import Autocomplete from '@mui/material/Autocomplete';
import CheckBoxOutlineBlankIcon from '@mui/icons-material/CheckBoxOutlineBlank';
import CheckBoxIcon from '@mui/icons-material/CheckBox';
import Checkbox from '@mui/material/Checkbox';
import { TextField } from '@mui/material';
import React,{useState} from 'react';
import 'bootstrap/dist/css/bootstrap.css';
const Algoselect=()=> {
  const icon = <CheckBoxOutlineBlankIcon fontSize="small" />;
const checkedIcon = <CheckBoxIcon fontSize="small" />;
const [selectedOptions, setSelectedOptions] = useState([]);
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
        console.log(value)
      setSelectedOptions(value[0]);
      }
      const handleSubmit = () => {
        console.log(selectedOptions[0][0]);
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
          <TextField {...params} label="choose algorithms" placeholder="enter your algorithm" />
        )}
      
     />
     <br/>
     <hr/>
    <button className='btn btn-primary' onClick={handleSubmit}>Submit!</button>
     </div>
     
    );
  }
export default Algoselect;