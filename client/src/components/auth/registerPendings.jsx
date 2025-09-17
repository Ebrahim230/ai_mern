import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import instance from '../../config/instance'
import logo from '../../assets/logo.png'
import './style.scss'

const RegisterPendings = ({ _id }) => {
  const navigate = useNavigate()

  const [formData, setFormData] = useState({
    fName: '',
    lName: ''
  })

  const formHandle = async (e) => {
    e.preventDefault()
    if (formData?.fName && formData?.lName) {
      let res = null
      try {
        // const token = document.cookie.split('; ').find(row => row.startsWith('userToken='));
        // const userToken = token ? token.split('=')[1] : null;
        // console.log('userToken:', userToken);

        // if(!userToken){
        //   alert('No valid user token found.');
        //   return 
        // }

        // Proceed with PUT request
        res = await instance.put('/api/user/signup-finish', {
          _id,
          fName: formData.fName,
          lName: formData.lName
        }
        // , 
        // {
        //   headers: {
        //     Authorization: `Bearer ${userToken}` 
        //   }
        // }
        
      )
      } catch (err) {
        console.log(err)
        if (err?.response?.data?.status === 422) {
          alert("Already Registered")
          navigate('/login')
        } 
      } finally {
        if (res?.data?.status === 200) { 
          navigate('/')
        } else if (res?.data?.status === 208) { 
          navigate('/login')
        }
      }
    } else {
      alert("Please enter your full name")
    }
  }

  return (
    <div className='Contain'>
      <div className='icon'>
        <img src={logo}/>
      </div>

      <h1>Tell us about you</h1>

      <form className="pendings" onSubmit={formHandle}>
        <div className="fullName">
          <input
            type="text"
            value={formData.fName}
            placeholder='First name'
            onChange={(e) => setFormData({ ...formData, fName: e.target.value })}
          />
          <input
            type="text"
            value={formData.lName}
            placeholder='Last name'
            onChange={(e) => setFormData({ ...formData, lName: e.target.value })}
          />
        </div>

        <button type='submit'>Continue</button>

        <div>
          <p>By clicking "Continue", you agree to our <span>Terms</span>, <br /><span>Privacy policy</span> and confirm you're 18 years or older.</p>
        </div>
      </form>
    </div>
  )
}

export default RegisterPendings
