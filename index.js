import { spawn } from 'node:child_process';

const memoryUsage = [[0.1, 0.2, 0.3, 0.4, 0.99, 0.99, 0.99, 0.99, 0.99, 0.9]];
const python = spawn('python', ['-u', 'hybrid_predict.py', JSON.stringify(memoryUsage)]);

python.stdout.on('data', (data) => {
    const [predictedMemory, isAnomaly] = data.toString().trim().split(',');
    console.log(`Predicted Memory js: ${predictedMemory}`);
    console.log(`Is Anomaly js: ${isAnomaly}`);
});

python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
});

python.on('close', (code) => {
    console.log(`Child process exited with code ${code}`);
});
