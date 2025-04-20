import axios from "axios";

// const podName = "finalpod-77c649c5fc-tzvnb"
const podName = "finalpod-64ff6f79c5-vdtgt"

const res = await axios.get(`http://localhost:8001/api/v1/namespaces/default/pods/${podName}`);
console.log(res)