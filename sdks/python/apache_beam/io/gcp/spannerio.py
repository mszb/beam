#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is experimental module for reading from Google Cloud Spanner.
https://cloud.google.com/spanner

To read from Cloud Spanner, apply ReadFromSpanner transformation.
It will return a PCollection of list, where each element represents an
individual row returned from the read operation.
Both Query and Read APIs are supported. See more information
about "https://cloud.google.com/spanner/docs/reads".

To execute a "query", specify a "ReadFromSpanner.with_query(QUERY_STRING)"
during the construction of the transform. For example:

    records = (pipeline | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME)
                  .with_query('SELECT * FROM users'))

To use the read API, specify a "ReadFromSpanner.with_table(TABLE_NAME, COLUMNS)"
during the construction of the transform. For example:

    records = (pipeline | ReadFromSpanner(PROJECT_ID, INSTANCE_ID, DB_NAME)
                  .with_table("users", ["id", "name", "email"]))

"ReadFromSpanner.with_table" also support indexes by specifying the "index"
parameter. For more information, the spanner read with index documentation:
https://cloud.google.com/spanner/docs/secondary-indexes#read-with-index


It is possible to read several PCollection of ReadOperation within a single
transaction. Apply ReadFromSpanner.create_transaction() transform, that lazily
creates a transaction. The result of this transformation can be passed to
read operation using ReadFromSpanner.with_transaction(). For Example:

    transaction = ReadFromSpanner.create_transaction(
        project_id=PROJECT_ID,
        instance_id=sINSTANCE_ID,
        database_id=DB_NAME,
        exact_staleness=datetime.timedelta(seconds=100))

    spanner_read = ReadFromSpanner(
        project_id=PROJECT_ID,
        instance_id=INSTANCE_ID,
        database_id=DB_NAME)

    users = (pipeline
            | 'Get all users' >> spanner_read.with_transaction(transaction)
               .with_query("SELECT * FROM users"))
    tweets = (pipeline
            | 'Get all tweets' >> spanner_read.with_transaction(transaction)
             .with_query("SELECT * FROM tweets"))
"""
from __future__ import absolute_import

import collections
import warnings

from google.cloud.spanner import Client
from google.cloud.spanner import KeySet
from google.cloud.spanner_v1.database import BatchSnapshot

import apache_beam as beam
from apache_beam.transforms import PTransform
from apache_beam.utils.annotations import experimental

##################################################
##########   WRITE      ##########################
##################################################
from google.cloud.spanner_v1.types import Mutation
from google.cloud.spanner_v1 import batch
import sys


__all__ = [
    'ReadFromSpanner', 'ReadOperation',
    'WriteToSpanner', 'WriteMutation', 'BatchFn', 'MutationGroup',
    '_BatchableFilterFn', '_WriteGroup'
]


class ReadOperation(collections.namedtuple("ReadOperation",
                                           ["read_operation", "batch_action",
                                            "transaction_action", "kwargs"])):
  """
  Encapsulates a spanner read operation.
  """

  __slots__ = ()

  @classmethod
  def with_query(cls, sql, params=None, param_types=None):
    return cls(
        read_operation="process_query_batch",
        batch_action="generate_query_batches", transaction_action="execute_sql",
        kwargs={'sql': sql, 'params': params, 'param_types': param_types}
    )

  @classmethod
  def with_table(cls, table, columns, index="", keyset=None):
    keyset = keyset or KeySet(all_=True)
    if not isinstance(keyset, KeySet):
      raise ValueError("keyset must be an instance of class "
                       "google.cloud.spanner_v1.keyset.KeySet")
    return cls(
        read_operation="process_read_batch",
        batch_action="generate_read_batches", transaction_action="read",
        kwargs={'table': table, 'columns': columns, 'index': index,
                'keyset': keyset}
    )


class _BeamSpannerConfiguration(collections.namedtuple(
    "_BeamSpannerConfiguration", ["project", "instance", "database",
                                  "credentials", "user_agent", "pool",
                                  "snapshot_read_timestamp",
                                  "snapshot_exact_staleness"])):

  @property
  def snapshot_options(self):
    snapshot_options = {}
    if self.snapshot_exact_staleness:
      snapshot_options['exact_staleness'] = self.snapshot_exact_staleness
    if self.snapshot_read_timestamp:
      snapshot_options['read_timestamp'] = self.snapshot_read_timestamp
    return snapshot_options


class ReadFromSpanner(object):

  def __init__(self, project_id, instance_id, database_id, pool=None,
               read_timestamp=None, exact_staleness=None, credentials=None,
               user_agent=None):
    """
    Read from Google Spanner.

    Args:
      project_id: The ID of the project which owns the instances, tables
        and data.
      instance_id: The ID of the instance.
      database_id: The ID of the database instance.
      user_agent: (Optional) The user agent to be used with API request.
      pool: (Optional) session pool to be used by database.
      read_timestamp: (Optional) Execute all reads at the given timestamp.
      exact_staleness: (Optional) Execute all reads at a timestamp that is
        ``exact_staleness`` old.
    """
    warnings.warn("ReadFromSpanner is experimental.", FutureWarning,
                  stacklevel=2)
    self._transaction = None
    self._options = _BeamSpannerConfiguration(
        project=project_id, instance=instance_id, database=database_id,
        credentials=credentials, user_agent=user_agent, pool=pool,
        snapshot_read_timestamp=read_timestamp,
        snapshot_exact_staleness=exact_staleness
    )

  def with_query(self, sql, params=None, param_types=None):
    read_operation = [ReadOperation.with_query(sql, params, param_types)]
    return self.read_all(read_operation)

  def with_table(self, table, columns, index="", keyset=None):
    read_operation = [ReadOperation.with_table(
        table=table, columns=columns, index=index, keyset=keyset
    )]
    return self.read_all(read_operation)

  def read_all(self, read_operations):
    if self._transaction is None:
      return _BatchRead(read_operations=read_operations,
                        spanner_configuration=self._options)
    else:
      return _NaiveSpannerRead(transaction=self._transaction,
                               read_operations=read_operations,
                               spanner_configuration=self._options)

  @staticmethod
  @experimental(extra_message="(ReadFromSpanner)")
  def create_transaction(project_id, instance_id, database_id, credentials=None,
                         user_agent=None, pool=None, read_timestamp=None,
                         exact_staleness=None):
    """
    Return the snapshot state for reuse in transaction.

    Args:
      project_id: The ID of the project which owns the instances, tables
        and data.
      instance_id: The ID of the instance.
      database_id: The ID of the database instance.
      credentials: (Optional) The OAuth2 Credentials to use for this client.
      user_agent: (Optional) The user agent to be used with API request.
      pool: (Optional) session pool to be used by database.
      read_timestamp: (Optional) Execute all reads at the given timestamp.
      exact_staleness: (Optional) Execute all reads at a timestamp that is
        ``exact_staleness`` old.
      """
    _snapshot_options = {}
    if read_timestamp:
      _snapshot_options['read_timestamp'] = read_timestamp
    if exact_staleness:
      _snapshot_options['exact_staleness'] = exact_staleness

    spanner_client = Client(project=project_id, credentials=credentials,
                            user_agent=user_agent)
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id, pool=pool)
    snapshot = database.batch_snapshot(**_snapshot_options)
    return snapshot.to_dict()

  def with_transaction(self, transaction):
    self._transaction = transaction
    return self


class _NaiveSpannerReadDoFn(beam.DoFn):

  def __init__(self, snapshot_dict, spanner_configuration):
    self._snapshot_dict = snapshot_dict
    self._spanner_configuration = spanner_configuration
    self._snapshot = None

  def to_runner_api_parameter(self, context):
    return self.to_runner_api_pickled(context)

  def setup(self):
    spanner_client = Client(self._spanner_configuration.project)
    instance = spanner_client.instance(self._spanner_configuration.instance)
    database = instance.database(self._spanner_configuration.database,
                                 pool=self._spanner_configuration.pool)
    self._snapshot = BatchSnapshot.from_dict(database, self._snapshot_dict)

  def process(self, element):
    with self._snapshot._get_session().transaction() as transaction:
      for row in getattr(transaction, element.transaction_action)(
          **element.kwargs):
        yield row

  def teardown(self):
    if self._snapshot:
      self._snapshot.close()


class _NaiveSpannerRead(PTransform):
  """
  A naive version of Spanner read that use transactions for read and execute
  sql methods from the previous state.
  """

  def __init__(self, transaction, read_operations, spanner_configuration):
    self._transaction = transaction
    self._read_operations = read_operations
    self._spanner_configuration = spanner_configuration

  def expand(self, pbegin):
    return (pbegin
            | 'Add Read Operations' >> beam.Create(self._read_operations)
            | 'Reshuffle' >> beam.Reshuffle()
            | 'Perform Read' >> beam.ParDo(
                _NaiveSpannerReadDoFn(
                    snapshot_dict=self._transaction,
                    spanner_configuration=self._spanner_configuration
                )))


class _BatchRead(PTransform):
  """
  This transform uses the Cloud Spanner BatchSnapshot to perform reads from
  multiple partitions.
  """

  def __init__(self, read_operations, spanner_configuration):

    if not isinstance(spanner_configuration, _BeamSpannerConfiguration):
      raise ValueError("spanner_configuration must be a valid "
                       "_BeamSpannerConfiguration object.")

    self._read_operations = read_operations
    self._spanner_configuration = spanner_configuration

  def expand(self, pbegin):
    spanner_client = Client(project=self._spanner_configuration.project,
                            credentials=self._spanner_configuration.credentials,
                            user_agent=self._spanner_configuration.user_agent)
    instance = spanner_client.instance(self._spanner_configuration.instance)
    database = instance.database(self._spanner_configuration.database,
                                 pool=self._spanner_configuration.pool)
    snapshot = database.batch_snapshot(**self._spanner_configuration
                                       .snapshot_options)

    reads = [
        {"read_operation": ro.read_operation, "partitions": p}
        for ro in self._read_operations
        for p in getattr(snapshot, ro.batch_action)(**ro.kwargs)
    ]

    return (pbegin
            | 'Generate Partitions' >> beam.Create(reads)
            | 'Reshuffle' >> beam.Reshuffle()
            | 'Read From Partitions' >> beam.ParDo(
                _ReadFromPartitionFn(
                    snapshot_dict=snapshot.to_dict(),
                    spanner_configuration=self._spanner_configuration)))


class _ReadFromPartitionFn(beam.DoFn):

  def __init__(self, snapshot_dict, spanner_configuration):
    self._snapshot_dict = snapshot_dict
    self._spanner_configuration = spanner_configuration

  def to_runner_api_parameter(self, context):
    return self.to_runner_api_pickled(context)

  def setup(self):
    spanner_client = Client(self._spanner_configuration.project)
    instance = spanner_client.instance(self._spanner_configuration.instance)
    self._database = instance.database(self._spanner_configuration.database,
                                       pool=self._spanner_configuration.pool)

  def process(self, element):
    self._snapshot = BatchSnapshot.from_dict(self._database,
                                             self._snapshot_dict)
    read_operation = element['read_operation']
    elem = element['partitions']

    for row in getattr(self._snapshot, read_operation)(elem):
      yield row

  def teardown(self):
    if self._snapshot:
      self._snapshot.close()


###############################################################################
###############################################################################
###############################################################################

from apache_beam.transforms import window
from builtins import list

xx_Mutator = collections.namedtuple('xx_Mutator', ["mutation", "operation"])

class _Mutator(collections.namedtuple('_Mutator', ["mutation", "operation"])):

  __slots__ = ()

  @property
  def byte_size(self):
    return self.mutation.ByteSize()


class MutationGroup(list):

  # todo: Check type safety. This should be list of _Mutator

  @property
  def byte_size(self):
    s = 0
    for m in self.__iter__():
      s += m.byte_size
    return s

  def primary(self):
    return self.__iter__().next()


class MutationSizeEstimator:

  def __init__(self, m):
    self._m = m

  @staticmethod
  def get_operation(m):
    pass


class WriteMutation:

  @staticmethod
  def insert(table, columns, values):
    """Insert one or more new table rows.
    :type table: str
    :param table: Name of the table to be modified.
    :type columns: list of str
    :param columns: Name of the table columns to be modified.
    :type values: list of lists
    :param values: Values to be modified.
    """
    return _Mutator(
        mutation=Mutation(insert=batch._make_write_pb(table, columns, values)),
        operation='insert'
    )



  @staticmethod
  def update(table, columns, values):
    """Update one or more existing table rows.
    :type table: str
    :param table: Name of the table to be modified.
    :type columns: list of str
    :param columns: Name of the table columns to be modified.
    :type values: list of lists
    :param values: Values to be modified.
    """
    return _Mutator(
        mutation=Mutation(update=batch._make_write_pb(table, columns, values)),
        operation='update')

  @staticmethod
  def insert_or_update(table, columns, values):
    """Insert/update one or more table rows.
    :type table: str
    :param table: Name of the table to be modified.
    :type columns: list of str
    :param columns: Name of the table columns to be modified.
    :type values: list of lists
    :param values: Values to be modified.
    """
    return _Mutator(
        mutation=Mutation(insert_or_update=batch._make_write_pb(table, columns, values)),
        operation='insert_or_update')

  @staticmethod
  def replace(table, columns, values):
    """Replace one or more table rows.
    :type table: str
    :param table: Name of the table to be modified.
    :type columns: list of str
    :param columns: Name of the table columns to be modified.
    :type values: list of lists
    :param values: Values to be modified.
    """
    return _Mutator(
        mutation=Mutation(replace=batch._make_write_pb(table, columns, values)),
        operation="replace"
    )

  @staticmethod
  def delete(table, keyset):
    """Delete one or more table rows.
    :type table: str
    :param table: Name of the table to be modified.
    :type keyset: :class:`~google.cloud.spanner_v1.keyset.Keyset`
    :param keyset: Keys/ranges identifying rows to delete.
    """
    delete = Mutation.Delete(table=table, key_set=keyset._to_pb())
    return _Mutator(
        mutation=Mutation(delete=delete),
        operation='delete'
    )


class WriteToSpanner(object):

  def __init__(self, project_id, instance_id, database_id):
    self._project_id = project_id
    self._instance_id = instance_id
    self._database_id = database_id

  def insert(self):
    return _Insert(self._project_id, self._instance_id, self._database_id)

  def batch(self):
    pass



class _BatchWrite(PTransform):

  def __init__(self, project_id, instance_id, database_id):
    self._project_id = project_id
    self._instance_id = instance_id
    self._database_id = database_id

  def expand(self, pcoll):
    return pcoll | beam.ParDo(_WriteToSpanner(
        self._project_id, self._instance_id, self._database_id
    ))



# @typehints.with_input_types(
#   typehints.Tuple[str, typehints.List[str],
#                   typehints.List[typehints.Tuple[T, ...]]])
class _Insert(PTransform):

  def __init__(self, project_id, instance_id, database_id):
    self._project_id = project_id
    self._instance_id = instance_id
    self._database_id = database_id

  def expand(self, pcoll):

    """
    SpannerIO.java:914
    // First, read the Cloud Spanner schema.
    // Build a set of Mutation groups from the current bundle,
      // sort them by table/key then split into batches.
    // Merge the batchable and unbatchable mutations and write to Spanner.
    """
    return pcoll | beam.ParDo(_WriteToSpanner(
        self._project_id, self._instance_id, self._database_id
    ))



class MutationSizeEstimator:

  def __init__(self):
    pass

  @staticmethod
  def get_operation(m):
    ops = ('insert', 'insert_or_update', 'replace', 'update', 'delete')
    for op in ops:
      if getattr(m, op).table is not None:
        return op
    return ValueError("Operation is not defined!")




class BatchFn(beam.DoFn):

  def __init__(self, max_batch_size_bytes, max_num_mutations, schema_view):
    self._max_batch_size_bytes = max_batch_size_bytes
    self._max_num_mutations = max_num_mutations
    self._schema_view = schema_view

  def start_bundle(self):
    print("start_bundle >>> ")
    self._batch = MutationGroup()
    self._size_in_bytes = 0L


  def process(self, element):
    batch_size_bytes = 0
    batch_cells = 0

    _max_bytes = 1024 * 1024 # 1 mb
    _max_bytes = self._max_batch_size_bytes

    mg = element
    mg_size = mg.byte_size

    if mg_size + self._size_in_bytes > _max_bytes:
      yield self._batch
      self._size_in_bytes = 0L
      # print("----------------- daz -----------")
      self._batch = MutationGroup()

    self._batch.extend(mg)
    self._size_in_bytes += mg_size
    # print(">>> ", self._size_in_bytes)

  def finish_bundle(self):
    # print("finish_bundle >>> ", len(self._batch))
    yield window.GlobalWindows.windowed_value(self._batch)
    self._batch = None


class _BatchableFilterFn(beam.DoFn):

  OUTPUT_TAG_UNBATCHABLE = 'unbatchable'

  def __init__(self, max_batch_size_bytes, max_num_mutations, schema_view):
    self._max_batch_size_bytes = max_batch_size_bytes
    self._max_num_mutations = max_num_mutations
    self._schema_view = schema_view
    self._batchable = None
    self._unbatchable = None

  def process(self, element):
    _max_bytes = self._max_batch_size_bytes
    mg = element
    mg_size = mg.byte_size
    if mg_size > _max_bytes:
      yield beam.pvalue.TaggedOutput(
          _BatchableFilterFn.OUTPUT_TAG_UNBATCHABLE,
          element)
    else:
      yield element


class _WriteToSpanner(beam.DoFn):

  def __init__(self, project_id, instance_id, database_id):
    self._project_id = project_id
    self._instance_id = instance_id
    self._database_id = database_id
    self._db_instance = None

  def to_runner_api_parameter(self, unused_context):
    pass

  def setup(self):
    spanner_client = Client(self._project_id)
    instance = spanner_client.instance(self._instance_id)
    self._db_instance = instance.database(self._database_id)


  def process(self, element):
    with self._db_instance.batch() as b:
      b._mutations.extend([x.mutation for x in element])

    return element

  # def process(self, element):
  #   _id = int(time.time())
  #   def _process(transaction):
  #     sql = "INSERT roles (key, rolename) VALUES ({}, 'insert-role-{}')".format(_id, _id)
  #     transaction.execute_update(sql)
  #   self._db_instance.run_in_transaction(_process)
  #   return element



class _WriteGroup(PTransform):

  def __init__(self, project_id, instance_id, database_id,
               max_batch_size_bytes=100L,
               max_num_mutations=None,
               schema_view=None):
    self._project_id = project_id
    self._instance_id = instance_id
    self._database_id = database_id

    self._max_batch_size_bytes = max_batch_size_bytes
    self._max_num_mutations = max_num_mutations
    self._schema_view = schema_view

  def expand(self, pcoll):
    filter_batchable_mutations = (
        pcoll
        | 'Filtering Batchable Murations' >> beam.ParDo(
        _BatchableFilterFn(self._max_batch_size_bytes, self._max_num_mutations,
                           self._schema_view)).with_outputs(
        _BatchableFilterFn.OUTPUT_TAG_UNBATCHABLE,
        main='batchable')
    )

    sorting = "" # todo: apply sorting

    batching_batchables = (
        filter_batchable_mutations['batchable']
        | beam.ParDo(BatchFn(self._max_batch_size_bytes, None, None))
    )

    return (
        (batching_batchables,
         filter_batchable_mutations[_BatchableFilterFn.OUTPUT_TAG_UNBATCHABLE])
      | 'Merging batchable and unbatchable' >> beam.Flatten()
    )

















