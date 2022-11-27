import React, { useState } from "react";
import ReactFileReader from 'react-file-reader';
import { CsvToHtmlTable } from 'react-csv-to-table';
import 'bootstrap/dist/css/bootstrap.css';
import axios from 'axios';
const Datadisplay=()=>{
    const [csv,setcsv]=useState('')
    const [file,setfile]=useState()
    const handleFiles = files => {
        var reader = new FileReader();
        setfile(reader)
        reader.onload = function(e) {
          setcsv(reader.result)
        }
      reader.readAsText(files[0]);
   }
   axios.post('http://127.0.0.1:5000/UpdateDateSet',{
    headers: {'Content-Type': 'multipart/form-data', 'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Allow-Credentials': 'true'}},file)
    .then(function (response) {
    console.log(response);
  })
  .catch(function (error) {
    console.log(error);
  });
   
    return <div className="container">
          <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        height: '50%',
        backgroundColor: '#fafafa',
        margin: '20px',
      }}>
        <h1 className='text'>Welcome to ML lab expreiments</h1>
        <ReactFileReader handleFiles={handleFiles} fileTypes={'.csv'}>
                <div><button className='btn btn-primary'>Upload your data set</button></div> 
          </ReactFileReader>
          <CsvToHtmlTable
               data={csv}
               csvDelimiter=","
               tableClassName="table thead-dark thead-light table-hover"
               style={{ margin: '20px'}}/>
               </div>
    </div>
}

export default Datadisplay;