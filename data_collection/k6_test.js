import http from "k6/http";
import { sleep } from "k6";

export const options = {
  scenarios: {
    cpu_scenario: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "1h", target: 8 },  // ramp up to 4 users
        { duration: "2h", target: 10 }, // stay at 10 users
        {duration:"30m", target: 15},
        { duration: "1h", target: 4 },
        { duration: "1h", target: 0 },  
      ],
      exec: "cpuTest", 
    },
    memory_scenario: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "1h", target: 6 },  // ramp up to 2 users
        { duration: "1h", target: 6 },  // hold at 6 users
        { duration: "1h", target: 2 },
        { duration: "30m", target: 10 },
        { duration: "1h", target: 5 },
        { duration: "1h", target: 0 },  // ramp down
      ],
      exec: "memoryTest", 
    },
  },
};

export function cpuTest() {
  http.get("http://localhost:3000/cpu");
  sleep(1);
}

export function memoryTest() {
  http.get("http://localhost:3000/memory");
  sleep(10);
}
