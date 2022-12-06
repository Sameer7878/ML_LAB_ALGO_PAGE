import Autocomplete from '@mui/material/Autocomplete';
import CheckBoxOutlineBlankIcon from '@mui/icons-material/CheckBoxOutlineBlank';
import CheckBoxIcon from '@mui/icons-material/CheckBox';
import Checkbox from '@mui/material/Checkbox';
import { TextField } from '@mui/material';
import React,{useState} from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import axios from 'axios';
import Getjsonresponse from './Getjsonresponse';
const Algoselect=(props)=> {
  const filename = props.name
const icon = <CheckBoxOutlineBlankIcon fontSize="small" />;
const checkedIcon = <CheckBoxIcon fontSize="small" />;
const [selectedOptions, setSelectedOptions] = useState([]);
const [options, setOptions] = useState([])
const [array, setArray] = useState([]);
    const algorithm = [
        { title: 'FIND-S algorithm', algo_no: 1},
        { title: 'Candidate-Elimination algorithm', algo_no:2 },
        { title: 'ID3 algorithm', algo_no:3 },
        { title: 'Backpropagation algorithm', algo_no:4 },
        { title: 'Gaussian Naive Bayesian', algo_no:5 },
        { title: "Multinomial Naive Bayesian", algo_no:6 },
        { title: 'Bernoulli Naive Bayesian', algo_no:7 },
        {title: 'Bayesian network', algo_no: 8},
        {title:'Gaussian Mixture Model(EM Algo)',algo_no:9},
        {title: 'K-means clustering', algo_no: 10}
        ];
        
      const handleChange = (event, value) => {
        const valu={...selectedOptions,value}
        setSelectedOptions(valu);

      }
      const handleSubmit = () => {
        const value = selectedOptions.value
        console.log(value)
        const algo_no = value.map((item) => item.algo_no)
        setArray(algo_no)
        function removeDuplicates(arr) {
                  return [...new Set(arr)];
        }
        const do_algo=removeDuplicates(array)
        axios.get(`http://127.0.0.1:5000/GetResultAsJson?filename=${filename}&do_algo=${array}`)
        .then((res)=>{
              console.log(res.data)
        })
        .catch((err)=>{
            console.log(err.message)})
      }
        
    return (<div>
      <Autocomplete
      onChange={handleChange}
      style={{alignItems:"center"}}
        multiple
        options={algorithm}
        disableCloseOnSelect
        getOptionLabel={(option) => option.title}
        renderOption={(props, option, { selected }) => (
          <li {...props}>
            <Checkbox
              icon={icon}
              checkedIcon={checkedIcon}
              style={{ marginRight: 8 }}
              checked={selected}
              onClick={()=>{console.log(option.title)}}
            />
            {option.title}
          </li>
        )}
        renderInput={(params) => (
          <TextField {...params} label="choose algorithms" placeholder="Enter your algorithm" />
        )}
      
     />
     <br/>
    <button className='btn btn-primary' onClick={handleSubmit}>Submit!</button>
    <div>
      {options.slice().map((e, index) => {
        return (
        <div key={index}>
      <p>{e.algo_no} {e.title} </p>
      
          
      </div>) 
      })}
    </div>
      <Getjsonresponse name={filename} array={array}/>
     </div>
     
    );
  }
export default Algoselect;