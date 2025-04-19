import axios from "axios";

const podName = "finalpod-77c649c5fc-tzvnb"
const res = await axios.get(`http://localhost:8001/api/v1/namespaces/default/pod/${podName}`);
console.log(res)