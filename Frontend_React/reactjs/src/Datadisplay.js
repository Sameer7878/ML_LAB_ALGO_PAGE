import React, { useState,useEffect } from "react";
import ReactFileReader from 'react-file-reader';
import { CsvToHtmlTable } from 'react-csv-to-table';
import 'bootstrap/dist/css/bootstrap.css';
import axios from 'axios';
const Datadisplay=()=>{
    const [csv,setcsv]=useState('')
    console.log(csv)
    const [file,setfile]=useState()
    const handleFiles = files => {
        var reader = new FileReader();
        setfile(reader)
        reader.onload = function(e) {
          setcsv(reader.result)
        }
      reader.readAsText(files[0]);
   }
useEffect(()=>{
  console.log(csv)
  var data = new FormData();
  data.append('file', csv.csv);

var config = {
  method: 'post',
  url: 'http://localhost:5000/upload',
  headers: {  'Content-Type': 'multipart/form-data' },
  data : data
};
  axios(config)
  .then(function (response) {
    console.log(JSON.stringify(response.data));
  })
  .catch(function (error) {
    console.log(error);
  });
},[csv])


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