import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { RegisterPendings, SignupComponent } from "../components";
import instance from "../config/instance";
import { setLoading } from "../redux/loading";
import "./style.scss";

const Signup = () => {
  const { user } = useSelector((state) => state);
  const { id } = useParams();
  const dispatch = useDispatch();
  const location = useLocation();
  const navigate = useNavigate();

  const [pending, setPending] = useState(false);

  useEffect(() => {
    dispatch(setLoading(true));

    const checkPending = async () => {
      try {
        const res = await instance.get("/api/user/checkPending", { params: { _id: id } });

        if (res?.data?.status === 208) {
          setPending(false);
        } else {
          setPending(true);
        }
      } catch (err) {
        console.error("Error checking pending status:", err);

        if (err?.response?.status === 404) {
          navigate("/404");
        } else {
          alert("An error occurred. Redirecting to signup...");
          navigate("/signup");
        }
      } finally {
        setTimeout(() => {
          dispatch(setLoading(false));
        }, 1000);
      }
    };

    if (!user) {
      if (location.pathname === "/signup") {
        setPending(false);
        setTimeout(() => dispatch(setLoading(false)), 1000);
      } else {
        checkPending();
      }
    }
  }, [user, id, location.pathname, dispatch, navigate]);

  return (
    <div className="Auth">
      <div className="inner">
        {pending ? (
          <RegisterPendings _id={id} />
        ) : (
          <>
            <SignupComponent />
            <div className="bottum">
              <div className="start">
                <a href="#" target="_blank" rel="noopener noreferrer">
                  Terms of use
                </a>
              </div>
              <div className="end">
                <a href="#" target="_blank" rel="noopener noreferrer">
                  Privacy Policy
                </a>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Signup;
