import React, { useState,useEffect } from "react";
import ReactFileReader from 'react-file-reader';
import { CsvToHtmlTable } from 'react-csv-to-table';
import 'bootstrap/dist/css/bootstrap.css';
import axios from "axios";
import Chooseoption from "./Chooseoption";
import Algoselect from "./Algoselect";
const Datadisplay=()=>{
    const [csv,setcsv]=useState('')
    const [filename,setfilename]=useState('')
    const [show, setShow] = useState(true);
    const handleFiles = files => {
      const data = new FormData();
      data.append('csv_file', files[0]);
      setfilename(files[0].name)
      console.log(filename)
      axios.post("http://127.0.0.1:5000/upload",data)
      .then((res)=>{
            console.log(res.data)
        })
      .catch((err)=>{
          console.log(err.message)})
        
      console.log(data)
      var reader = new FileReader();
      reader.onload = function(e) {
      // Use reader.result
       setcsv(reader.result)
      }
    reader.readAsText(files[0]);
 }
 const handleshow =() =>{
  setShow(prev => ! prev)
 }

 
    return (
     <div className="container" style={{backgroundColor: "#e5f2fa"}} >
          <ReactFileReader handleFiles={handleFiles} fileTypes={'.csv'}>
                  <br/>
                  <div style={{textAlign: "center"}}>
                  <h1 >ML lab expreiments</h1>
                  <hr/>
                  <button  className='btn btn-primary'>Upload</button>
                  </div>
           </ReactFileReader>
       <br/>
       <div style={{textAlign: "center"}}> <button  className="btn btn-primary" onClick={handleshow}>Hide Dataset</button>
       <br/>
      <br/>
        <Algoselect/>
       
       {show && <CsvToHtmlTable data={csv} csvDelimiter="," tableClassName="table  table-striped table-hover"/> }
         <Algoselect name ={filename}/>
    </div>
    </div>
 
     )
   }



export default Datadisplay;