import express, { response } from "express";
import axios from "axios";
import { exec } from "child_process";
import fs from "fs";
import { createObjectCsvWriter as createCsvWriter } from "csv-writer";

const app = express();
const PORT = 3001;
const PROMETHEUS_URL = "http://localhost:9090/api/v1/query";

const csvWriter = createCsvWriter({
  path: "k8s_metrics.csv",
  header: [
    { id: "podName", title: "Pod Name" },
    { id: "timestamp", title: "Timestamp" },
    { id: "node_status", title: "Node Status" },
    { id: "memory_usage", title: "Memory Usage" },
    { id: "cpu_load", title: "CPU Load" },
    { id: "restart_count", title: "Restart Count" },
    { id: "requests_duration", title: "Requests Duration" },
    { id: "source_time", title: "Source Time" },
    { id: "response_time", title: "Request Total" },
    { id: "request_errors", title: "Request Errors" }
  ]
});

// Get pod status from Prometheus (e.g., using kube_pod_status_phase)

async function getDetailedPodStatus(podName) {
  const res = await axios.get(`http://localhost:8001/api/v1/namespaces/default/pods/${podName}`);
  const podData = res.data;
  // console.log(podData.status.containerStatuses)

  const status = podData?.status?.containerStatuses[0]?.state?.terminated?.reason || podData?.status?.containerStatuses[0]?.state?.waiting?.reason || podData?.status?.phase;
  //const status = podData?.status?.containerStatuses?.[0]?.state;
  return status;
}

// Get restart count from Prometheus (e.g., using kube_pod_container_status_restarts_total)
async function getRestartCount(podName) {
  // Sum all containers' restart counts for the pod
  const value = await queryPrometheus(`sum(kube_pod_container_status_restarts_total{pod=\"${podName}\"})`);
  return value || 0;
}

async function queryPrometheus(query, timestamp = null) {
  try {
    const timeParam = timestamp ? `&time=${timestamp}` : "";
    const response = await axios.get(`${PROMETHEUS_URL}?query=${encodeURIComponent(query)}${timeParam}`);
    // return response.data.data.result.length > 0 ? response.data.data.result[0].value[1] : null;
    return response.data.data.result.length > 0 ? parseFloat(response.data.data.result[0].value[1]) : 0;
  } catch (error) {
    console.log("error", query)
    return 0;
  }
}

async function getK8sMetrics(podName) {
  try {

    const memory_usage = await queryPrometheus(
      `container_memory_usage_bytes{pod="${podName}"}`
    ) / 1000000;

    const cpu_load = await queryPrometheus(
      `rate(container_cpu_usage_seconds_total{pod="${podName}"}[1m]) * 1000`
    );

    return [memory_usage, cpu_load];
  } catch (error) {
    console.log('Error fetching Kubernetes metrics:', error);
    return [0, 0];
  }
}

async function collectMetrics() {
  const timestamp = new Date().toISOString();
  const podName = "server-image-6d6dc86df-jz8m5"

  // Get metrics from Kubernetes Metrics Server
  let [memory_usage, cpu_load] = await getK8sMetrics(podName);

  let [node_status, restart_count, requests_duration, source_time, response_time, request_errors] = await Promise.all([
    getDetailedPodStatus(podName),
    getRestartCount(podName),

    queryPrometheus(`histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{pod="${podName}",response_code="200"}[20s])) by (le))`) || 0,

    queryPrometheus(`(rate(istio_request_duration_milliseconds_sum{reporter="source",response_code="200",source_workload="istio-ingressgateway"}[20s])/ rate(istio_request_duration_milliseconds_count{reporter="source",response_code="200",source_workload="istio-ingressgateway"}[20s]))`) || 0,
    // queryPrometheus(`(rate(istio_request_duration_milliseconds_sum{reporter="destination",response_code="200",source_workload="istio-ingressgateway"}[1m])/rate(istio_request_duration_milliseconds_count{reporter="destination",response_code="200",source_workload="istio-ingressgateway"}[1m]))`) || 0,

    queryPrometheus(`sum(istio_requests_total{pod="${podName}"})`) || 0,
    queryPrometheus(`sum(increase(istio_requests_total{pod="${podName}",response_code!="200"}[20s]))`) || 0,
  ])

  let row = { podName, timestamp, node_status, memory_usage, cpu_load, restart_count, requests_duration, source_time, response_time, request_errors };
  await csvWriter.writeRecords([row]);

}

app.listen(PORT, async () => {
  await collectMetrics();
  setInterval(async () => {
    await collectMetrics();
  }, 1000);
  console.log(`Server running on http://localhost:${PORT}`);
});
