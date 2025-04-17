import express from "express";
import axios from "axios";
import { spawn, fork } from 'node:child_process';
import { Worker } from 'node:worker_threads'

const PROMETHEUS_URL = "http://localhost:9090/api/v1/query";
const app = express();
const PORT = 3001;

async function queryPrometheus(query, timestamp = null) {
    try {
        const timeParam = timestamp ? `&time=${timestamp}` : "";
        const response = await axios.get(`${PROMETHEUS_URL}?query=${encodeURIComponent(query)}${timeParam}`);
        return response.data.data.result.length > 0 ? response.data.data.result[0].value[1] : null;
    } catch (error) {
        return null;
    }
}

const podName = "finalpod-77c649c5fc-tzvnb"
let isRunning = false;
const memoryUsage = [];

const firstInterval = setInterval(async () => {
    if (isRunning) return;
    isRunning = true;
    if (memoryUsage.length < 10) {
        const pod_status = await queryPrometheus(`kube_pod_status_phase(pod="${podName}")`);
        const memory = await queryPrometheus(`container_memory_usage_bytes{pod="${podName}"}`) || 0;
        memoryUsage.push(memory);
        console.log(memoryUsage.length, memoryUsage);
    }

    if (memoryUsage.length === 10) {
        clearInterval(firstInterval);
        console.log("cleared");
    }

    isRunning = false;
}, 1000);


setTimeout(() => {
    const python = spawn('python', ['-u', 'hybrid_predict.py']);
    console.log("Python process spawned");

    // Attach listeners ONCE
    python.stdout.on('data', (data) => {
        const lines = data.toString().split('\n');
        for (const line of lines) {
            if (line.trim()) {
                const [predictedMemory, isAnomaly] = line.trim().split(',');
                console.log(`Predicted Memory js: ${predictedMemory}`);
                console.log(`Is Anomaly js: ${isAnomaly}`);
            }
        }
    });

    python.stderr.on('data', (data) => {
        console.error("Python STDERR:", data.toString());
    });

    python.stderr.on('error', (error) => {
        console.error("Python error:", error);
    });

    // Keep streaming input
    setInterval(async () => {
        const memory = await queryPrometheus(`container_memory_usage_bytes{pod="${podName}"}`) || 0;
        memoryUsage.shift();
        memoryUsage.push(memory);

        python.stdin.write(JSON.stringify({ podName: podName, memoryUsage: memoryUsage }) + '\n');
    }, 1000);
}, 11000);
