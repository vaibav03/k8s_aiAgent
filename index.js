import express from "express";
import axios from "axios";
import { spawn, exec } from 'node:child_process';
import {processAnomaly,} from './llm.js'

import { Worker } from 'node:worker_threads'
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { systemPrompt } from './prompt.js';

import 'dotenv/config'

// async function sendToLLM(prompt) {
//     const chat = new ChatGoogleGenerativeAI({
//         apiKey: process.env.GOOGLE_API_KEY,
//         model: "gemini-1.5-pro",
//         temperature: 0,
//         maxRetries: 2,
//     });

//     try {
//         const res = await chat.invoke([
//             { role: "system", content: systemPrompt },
//             { role: "user", content: prompt }
//         ]);
//         console.log(res.content);
//         return res.content;
//     } catch (error) {
//         console.log("API error",error.message);
//         throw error;
//     }
// }

function runShellCommand(cmd) {
    exec(cmd, (err, stdout, stderr) => {
        if (err) {
            console.error(`Shell error: ${stderr}`);
        } else {
            console.log(`Shell output:\n${stdout}`);
        }
    });
}

async function isPodCrashLooping(podName) {
    const res = await axios.get(`http://localhost:8001/api/v1/namespaces/default/pods/${podName}`);
    const podData = res.data;
    const status = podData?.status?.containerStatuses?.[0]?.state;
    return status?.waiting?.reason === 'CrashLoopBackOff';
}


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

const podName = "finalpod-64ff6f79c5-vdtgt"
let isRunning = false;
let memoryUsage = [];
// for(let i=0;i<=10;i++){
//     memoryUsage.push(1e6);
// }

async function getMaxLimit() {
    const result = await axios.get(`http://localhost:8001/api/v1/namespaces/default/pods/${podName}`);
    const podData = result.data;
    const limit = podData?.spec?.containers?.[0].resources?.limits?.memory;
    return parseInt(limit.split("M")[0])
}

let maxLimit = await getMaxLimit()

const firstInterval = setInterval(async () => {
    if (isRunning) return;
    isRunning = true;
    if (memoryUsage.length < 10) {
        // const pod_status = await queryPrometheus(`kube_pod_status_phase(pod="${podName}")`);
        const memory = await queryPrometheus(`container_memory_usage_bytes{pod="${podName}"}`) || 0;
        memoryUsage.push(memory);
    }

    if (memoryUsage.length === 10) {
        clearInterval(firstInterval);
        console.log("cleared");
        memoryUsage = memoryUsage.map((mem)=>{
            mem/=(1024*1024);
            mem/=(maxLimit);
            return mem;
        })
        console.log(memoryUsage)
    }
    isRunning = false;
}, 1000);


setTimeout(() => {
    const python = spawn('python', ['-u', 'hybrid_predict.py']);
    console.log("Python process spawned");

    // Attach listeners ONCE
    python.stdout.on('data', async (data) => {
        const lines = data.toString().split('\n');
        console.log(data)
        for (const line of lines) {
            if (line.trim()) {
                const [predictedMemory, isAnomaly_from_python] = line.trim().split(',');
                const isAnamoly =  parseInt(isAnomaly_from_python)
                console.log(`Predicted Memory js: ${predictedMemory}`);
                console.log(`Is Anomaly js: ${isAnamoly} and type ${typeof isAnamoly}`);

                if ( typeof isAnamoly !=="undefined" && isAnamoly == 1 ) {
                    const critical = await isPodCrashLooping(podName);

                    if (critical) {
                        console.log(`⚠️ CRITICAL: Pod ${podName} is leaking memory. Notify cluster administrator.`);
                    } else {
                        // const prompt = `The pod ${podName} is leaking memory, and here is the predicted memory usage: ${predictedMemory}. Suggest a shell command to rectify the issue. Here is the maximum alloted memory usage limit ${maxLimit}`;
                        // const shellCommand = await sendToLLM(prompt);
                        // console.log(`🧠 LLM suggests: ${shellCommand}`);
                        /// runShellCommand(shellCommand);
                        
                        try {
                            const recommendation_from_llm = await processAnomaly(podName,predictedMemory,maxLimit)
                            console.log("recommendation_from_llm",recommendation_from_llm)
                            if (recommendation_from_llm) {
                                // await runShellCommand(recommendation_from_llm.command.command)
                                console.log(recommendation_from_llm.command.command)
                            }
                        } catch(error) {
                            console.log("Error while calling llm",error.message)
                        }

                        maxLimit = await getMaxLimit();
                    }
                } 
                
                else {
                    console.log(`✅ Healthy. Predicted Memory: ${predictedMemory}`);
                }
            }
        }
    }
    );

    setInterval(async () => {
        let memory = await queryPrometheus(`container_memory_usage_bytes{pod="${podName}"}`) || 0;
        memory/= (1024*1024);
        memory/=maxLimit;
        memoryUsage.shift();
       // console.log(memoryUsage,"shifted")
        memoryUsage.push(memory);
        console.log(memoryUsage,"updated mem")

        python.stdin.write(JSON.stringify({ podName: podName, memoryUsage: memoryUsage }) + '\n');
    }, 1000);

}, 11000);
