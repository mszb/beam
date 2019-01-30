/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.beam.fn.harness.data;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.beam.fn.harness.SimpleExecutionState;
import org.apache.beam.model.fnexecution.v1.BeamFnApi.MonitoringInfo;
import org.apache.beam.runners.core.metrics.ExecutionStateTracker;
import org.apache.beam.runners.core.metrics.MetricsContainerImpl;
import org.apache.beam.runners.core.metrics.MetricsContainerStepMap;
import org.apache.beam.runners.core.metrics.SimpleMonitoringInfoBuilder;
import org.apache.beam.sdk.fn.function.ThrowingRunnable;
import org.apache.beam.sdk.metrics.MetricsEnvironment;

/**
 * A class to to register and retrieve functions for bundle processing (i.e. the start, or finish
 * function). The purpose of this class is to wrap these functions with instrumentation for metrics
 * and other telemetry collection.
 *
 * <p>Usage: // Instantiate and use the registry for each class of functions. i.e. start. finish.
 *
 * <pre>
 * PTransformFunctionRegistry startFunctionRegistry;
 * PTransformFunctionRegistry finishFunctionRegistry;
 * startFunctionRegistry.register(myStartThrowingRunnable);
 * finishFunctionRegistry.register(myFinishThrowingRunnable);
 *
 * // Then invoke the functions by iterating over them, in your desired order: i.e.
 * for (ThrowingRunnable startFunction : startFunctionRegistry.getFunctions()) {
 *   startFunction.run();
 * }
 *
 * for (ThrowingRunnable finishFunction : Lists.reverse(finishFunctionRegistry.getFunctions())) {
 *   finishFunction.run();
 * }
 * // Note: this is used in ProcessBundleHandler.
 * </pre>
 */
public class PTransformFunctionRegistry {

  private MetricsContainerStepMap metricsContainerRegistry;
  private ExecutionStateTracker stateTracker;
  private String executionTimeUrn;
  private List<ThrowingRunnable> runnables = new ArrayList<>();
  private List<SimpleExecutionState> executionStates = new ArrayList<SimpleExecutionState>();


  /**
   * Construct the registry to run for either start or finish bundle functions.
   * @param metricsContainerRegistry - Used to enable a metric container to properly account for
   *     the pTransform in user metrics.
   * @param stateTracker - The tracker to enter states in order to calculate execution time metrics.
   * @param executionTimeUrn - The URN to use for the execution time metrics.
   */
  public PTransformFunctionRegistry(
      MetricsContainerStepMap metricsContainerRegistry,
      ExecutionStateTracker stateTracker,
      String executionTimeUrn) {
    this.metricsContainerRegistry = metricsContainerRegistry;
    this.executionTimeUrn = executionTimeUrn;
    this.stateTracker = stateTracker;
  }

  /**
   * Register the runnable to process the specific pTransformId and track its execution time
   *
   * @param pTransformId
   * @param runnable
   */
  public void register(String pTransformId, ThrowingRunnable runnable) {
    HashMap<String, String> labelsMetadata = new HashMap<String, String>();
    labelsMetadata.put(SimpleMonitoringInfoBuilder.PTRANSFORM_LABEL, pTransformId);
    SimpleExecutionState state = new SimpleExecutionState(this.executionTimeUrn, labelsMetadata);
    executionStates.add(state);

    ThrowingRunnable wrapped =
        () -> {
          MetricsContainerImpl container = metricsContainerRegistry.getContainer(pTransformId);
          try (Closeable metricCloseable = MetricsEnvironment.scopedMetricsContainer(container)) {
            try (Closeable trackerCloseable = this.stateTracker.enterState(state)) {
              runnable.run();
            }
          }
        };
    runnables.add(wrapped);
  }

  // TODO write unit tests for this method and entering the state.
  public List<MonitoringInfo> getExecutionTimeMonitoringInfos() {
    List<MonitoringInfo> monitoringInfos = new ArrayList<MonitoringInfo>();
    for (SimpleExecutionState state : executionStates) {
      SimpleMonitoringInfoBuilder builder = new SimpleMonitoringInfoBuilder();
      builder.setUrn(this.executionTimeUrn);
      for (Map.Entry<String, String> entry : state.getLabels().entrySet()) {
        builder.setLabel(entry.getKey(), entry.getValue());
      }
      builder.setInt64Value(state.getTotalMillis());
      monitoringInfos.add(builder.build());
    }
    return monitoringInfos;
  }

  /**
   * @return A list of wrapper functions which will invoke the registered functions indirectly. The
   *     order of registry is maintained.
   */
  public List<ThrowingRunnable> getFunctions() {
    return runnables;
  }
}
