import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import FormFeild from './FormField';
import instance from '../../config/instance';
import logo from "../../assets/logo.png";
import './style.scss';

const ForgotComponent = ({ isRequest, token }) => {
  const [email, setEmail] = useState('');
  const [newPass, setNewPass] = useState('');
  const [reEnter, setReEnter] = useState('');
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleEmailChange = (e) => setEmail(e.target.value);
  const handlePassChange = (e) => setNewPass(e.target.value);
  const handleReEnterChange = (e) => setReEnter(e.target.value);

  const formHandleMail = async (e) => {
    e.preventDefault();
    try {
      const res = await instance.post('/api/user/forgot-request', { email });
      alert(res.data.message); // "If the email exists, a reset link has been sent."
    } catch (err) {
      setError('Error sending reset email');
    }
  };

  const formHandleReset = async (e) => {
    e.preventDefault();
    if (newPass.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    if (newPass !== reEnter) {
      setError('Passwords do not match');
      return;
    }
    try {
      const res = await instance.put('/api/user/forgot-finish', { token, newPass, reEnter });
      if (res.data.status === 200) {
        navigate('/login');
      } else {
        setError(res.data.message);
      }
    } catch (err) {
      setError('Error resetting password');
    }
  };

  return (
    <div className='Contain'>
      <div className='icon'>
        <img src={logo} alt="Logo" />
      </div>

      {isRequest ? (
        <form className='Form' onSubmit={formHandleMail}>
          <div>
            <div className="emailForgot">
              <FormFeild
                value={email}
                name={'email'}
                label={"Email address"}
                type={"email"}
                handleInput={handleEmailChange}
              />
            </div>
            <button type='submit'>Continue</button>
          </div>
        </form>
      ) : (
        <form className='Form' onSubmit={formHandleReset}>
          <div>
            <div className="password">
              <FormFeild
                value={newPass}
                name={'newPass'}
                label={"New password"}
                type={"password"}
                handleInput={handlePassChange}
              />
            </div>
            <div className="password">
              <FormFeild
                value={reEnter}
                name={'reEnter'}
                label={"Re-enter new password"}
                type={"password"}
                handleInput={handleReEnterChange}
              />
            </div>
            {error && <div className='error'>{error}</div>}
            <button type='submit'>Reset password</button>
          </div>
        </form>
      )}
    </div>
  );
};

export default ForgotComponent;