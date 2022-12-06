import React, { useEffect, useState } from "react";
import axios from "axios";
const Getjsonresponse = (props) => {
    const filename=props.name
    const array = props.array
    const [res,setres]=useState()
    useEffect(()=>{
        axios.get(`http://127.0.0.1:5000/GetResultAsJson?filename=${filename}&do_algo=${array}`)
        .then((res)=>{
            const result = res.data
            setres(result)
            console.log(res.data)
        })
        .catch((err)=>{
            console.log(err.message)
        })
    },[])
    return(<div>
        <h1>hello</h1>
        <p>{filename}</p>
        {array.map((e)=>{
            return <p>{e}</p>
        })}
        {res.map((e,i)=>{
            return<h1>{e}</h1>
        })}
    </div>)

}

export default Getjsonresponse;