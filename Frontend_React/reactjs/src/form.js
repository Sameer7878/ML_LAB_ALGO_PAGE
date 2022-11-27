import React, { useState } from 'react';
import axios from 'axios';


const Form = () => {
    const [csvFile, setCSVFile] = useState();
   
   
const handleFile = (e) => {
    const data = new FormData();
    data.append('file', e.target.files[0]);
    setCSVFile(data)

}

const handleSubmit = (e) => {
    e.preventDefault();
    // const data = new FormData();
    // data.append('file', this.uploadInput.files[0]);
    // console.log
    // console.log(csvFile)
    axios.post('http://localhost:5000/UpdateDateSet',csvFile)
    .then((res)=>{
        console.log(res)
    })
    .catch((err)=>{
        console.log(err.message)
    })
  
 }
    return (
    <div>
        <form onSubmit={handleSubmit}>
            <input type='file' accept='.csv' onChange={handleFile}/>
            <br />
            <button type='submit'> Submit </button>
           
        </form>
    </div>
);
}

export default Form;