import React, { Fragment, useCallback, useReducer, useState } from "react";
import { Tick, Mail } from "../../assets";
import { Link, useNavigate } from "react-router-dom";
import FormField from "./FormField";
import logo from "../../assets/logo.png";
import instance from "../../config/instance";
import "./style.scss";

const reducer = (state, { type, status }) => {
  switch (type) {
    case "filled":
      return { ...state, filled: status };
    case "error":
      return { ...state, error: status };
    case "mail":
      return { ...state, mail: status, error: !status };
    default:
      return state;
  }
};

const SignupComponent = () => {
  const navigate = useNavigate();
  const [state, stateAction] = useReducer(reducer, {
    filled: false,
    error: false,
    mail: false,
  });

  const [formData, setFormData] = useState({
    email: "",
    pass: "",
    manual: false,
  });

  const [isResending, setIsResending] = useState(false);

  const handleInput = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const formHandle = async (e) => {
    if (e) e.preventDefault();

    // Validate password length
    if (!formData.pass || formData.pass.length < 8) {
      alert("Password must be at least 8 characters long!");
      return;
    }

    try {
      const res = await instance.post("/api/user/signup", formData);
      console.log("Signup response:", res.data);

      if (res.status === 201) {
        stateAction({ type: "mail", status: true });
      } else if (res?.data?.data?.manual) {
        stateAction({ type: "mail", status: true }); 
      } else if (res?.data?.data?._id) {
        navigate(`/signup/pending/${res.data.data._id}`); 
      }
    } catch (err) {
      console.error("Signup error:", err.response?.data?.message || err.message);

      stateAction({ type: "error", status: true });
      alert(err.response?.data?.message || "Signup failed. Please try again.");
    }
  };

  const resendMail = async () => {
    setIsResending(true);
    await formHandle(null); 
    setTimeout(() => setIsResending(false), 2000); 
  };

  const passwordClass = useCallback((remove, add) => {
    document.querySelector(remove)?.classList?.remove("active");
    document.querySelector(add)?.classList?.add("active");
  }, []);

  return (
    <div className="Contain">
      <div className="icon">
        <img src={logo} alt="Logo" />
      </div>

      {!state.mail ? (
        <Fragment>
          <div>
            <h1>Create your account</h1>
            <p>
              Please note that phone verification is required for signup. Your number
              will only be used to verify your identity for security purposes.
            </p>
          </div>

          {!state.filled ? (
            <div className="options">
              <form
                className="manual"
                onSubmit={(e) => {
                  e.preventDefault();
                  setFormData((prev) => ({ ...prev, manual: true }));
                  stateAction({ type: "filled", status: true });
                }}
              >
                <div>
                  <FormField
                    value={formData.email}
                    name="email"
                    label="Email address"
                    type="email"
                    handleInput={handleInput}
                  />
                </div>
                <div>
                  <button type="submit">Continue</button>
                </div>
              </form>

              <div data-for="acc-sign-up-login">
                <span>Already have an account?</span>
                <Link to="/login/auth">Log in</Link>
              </div>
            </div>
          ) : (
            <form className="Form" onSubmit={formHandle}>
              <div>
                <div className="email">
                  <button
                    type="button"
                    onClick={() => stateAction({ type: "filled", status: false })}
                  >
                    Edit
                  </button>

                  <FormField
                    value={formData.email}
                    name="email"
                    type="email"
                    isDisabled
                    error={state.error}
                  />
                </div>

                {state.error && (
                  <div className="error">
                    <div>!</div> The user already exists.
                  </div>
                )}

                <div className="password">
                  <FormField
                    value={formData.pass}
                    name="pass"
                    label="Password"
                    type="password"
                    passwordClass={passwordClass}
                    handleInput={handleInput}
                  />
                </div>

                <div id="alertBox">
                  Your password must contain:
                  <p id="passAlertError" className="active">
                    <span>&#x2022;</span> At least 8 characters
                  </p>
                  <p id="passAlertDone" className="active">
                    <span>
                      <Tick /> 
                    </span>{" "}
                    At least 8 characters
                  </p>
                </div>

                <button type="submit">Continue</button>
              </div>

              <div data-for="acc-sign-up-login">
                <span>Already have an account?</span>
                <Link to="/login/auth">Log in</Link>
              </div>
            </form>
          )}
        </Fragment>
      ) : (
        <div className="mail">
          <div className="icon">
            <Mail />
          </div>

          <div>
            <h3>Check Your Email</h3>
          </div>

          <div>
            <p>
              Please check the email address {formData.email} for instructions to
              complete signup.
            </p>
          </div>

          <button onClick={resendMail} disabled={isResending}>
            {isResending ? "Resending..." : "Resend Mail"}
          </button>
        </div>
      )}
    </div>
  );
};

export default SignupComponent;
