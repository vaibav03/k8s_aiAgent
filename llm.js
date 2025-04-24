// llmService.js
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { systemPrompt } from "./prompt.js";
import 'dotenv/config';

// Rate limiting configuration
const RATE_LIMIT_WINDOW = 60000; // 1 minute in milliseconds
const MAX_REQUESTS_PER_WINDOW = 2;
const ANOMALY_THRESHOLD = 3; // Number of consecutive anomalies before calling LLM
const ANOMALY_RESET_WINDOW = 60000; // 1 minutes in milliseconds

class RateLimiter {
    constructor(limit = MAX_REQUESTS_PER_WINDOW) {
        this.requests = [];
        this.limit = limit;
        this.window = RATE_LIMIT_WINDOW;
    }

    async acquire() {
        const now = Date.now();
        this.requests = this.requests.filter(time => now - time < this.window);
        
        if (this.requests.length >= this.limit) {
            const oldest = this.requests[0];
            const waitTime = this.window - (now - oldest);
            await new Promise(resolve => setTimeout(resolve, waitTime));
        }
        
        this.requests.push(now);
    }
}

const rateLimiter = new RateLimiter();

// State to track anomalies
let anomalyCounter = 0;
let lastAnomalyTime = 0;

export async function processAnomaly(podName, predictedMemory,maxLimit) {
    // Reset counter if no anomalies for a while
    const now = Date.now();
    if (now - lastAnomalyTime > ANOMALY_RESET_WINDOW) {
        anomalyCounter = 0;
    }

    anomalyCounter++;
    lastAnomalyTime = now;

    // Only call LLM if we have multiple consecutive anomalies
    if (anomalyCounter >= ANOMALY_THRESHOLD) {
        try {
            await rateLimiter.acquire();
            const prompt = `
            Analyze memory usage for pod ${podName}:
            - predicted memory usage: ${predictedMemory}
            - Number of consecutive anomalies: ${anomalyCounter}
            - The current maximum alloted memory usage limit ${maxLimit}
            - Last ${ANOMALY_THRESHOLD} anomalies occurred within ${ANOMALY_RESET_WINDOW/1000/60} minutes
            `;
            
            console.log("calling llm")
            const response = await sendToLLM(prompt);
            const recommendation = JSON.parse(response);
            return recommendation;
        } catch (error) {
            console.error('Error processing anomaly:', error.message);
            throw error;
        }
    }
    return null; // Return null if not enough anomalies
}

export async function sendToLLM(prompt) {
    const chat = new ChatGoogleGenerativeAI({
        apiKey: process.env.GOOGLE_API_KEY,
        model: "gemini-1.5-pro",
        temperature: 0,
        maxRetries: 2,
    });

    try {
        const res = await chat.invoke([
            { role: "system", content: systemPrompt },
            { role: "user", content: prompt }
        ]);

        let content = res.content;
        
        content = content.replace(/```json\n?|```/g, ''); // Remove code block markers
        content = content.trim(); // Remove any extra whitespace
        
        console.log("llm is called with prompt",prompt,"and it returns(cleaned) ", content);
        return content;

    } catch (error) {
        console.log("API error", error.message);
        throw error;
    }
}

export function runShellCommand(cmd) {
    return new Promise((resolve, reject) => {
        exec(cmd, (err, stdout, stderr) => {
            if (err) {
                console.error(`Shell error: ${stderr}`);
                reject(err);
            } else {
                console.log(`Shell output:\n${stdout}`);
                resolve(stdout);
            }
        });
    });
}

// Export the prompt as well
// export { systemPrompt } from './prompts.js';