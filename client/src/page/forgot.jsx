import React from "react";
import { useSearchParams } from "react-router-dom";
import { ForgotComponent } from "../components";
import "./style.scss";

const Forgot = () => {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const isRequest = !token; // If token is present, it's a password reset request

  return (
    <div className="Auth">
      <div className="inner">
        <ForgotComponent
          isRequest={isRequest}
          token={token}
        />

        <div className="bottum">
          <div className="start">
            <a href="#terms of use" target="_blank">
              Terms of use
            </a>
          </div>
          <div className="end">
            <a href="#privacy policy" target="_blank">
              Privacy Policy
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Forgot;