import React, { useLayoutEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { setLoading } from "./redux/loading";
import instance from "./config/instance";
import { emptyUser, insertUser } from "./redux/user";
import { emptyAllRes } from "./redux/messages";

const ProtectedRoute = ({ offline, authed }) => {
  const [component, setComponent] = useState(<Outlet />); 
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();

  useLayoutEffect(() => {
    if (offline) {
      setComponent(<Outlet />);
      return;
    }

    dispatch(setLoading(true));

    const getResponse = async () => {
      let res = null;
      try {
        res = await instance.get("/api/user/checkLogged");
        if (res?.data?.data) {
          dispatch(insertUser(res.data.data));
          setComponent(<Outlet />); 
        }
      } catch (err) {
        console.error(err);
        
        if (err?.response?.data?.status === 200) {
          dispatch(emptyUser());
          dispatch(emptyAllRes());
          if (authed) {
            navigate("/login");
          } else {
            setComponent(<Outlet />);
          }
        } else if (err?.code !== "ERR_NETWORK") {
          navigate("/something-went-wrong");
        }
      } finally {
        dispatch(setLoading(false));
        if (res?.data?.status === 208) {
          if (!authed) {
            navigate("/");
          } else {
            setComponent(<Outlet />);
          }
        }
      }
    };

    getResponse();
  }, [location, authed, offline, navigate, dispatch]);

  return component;
};

export default ProtectedRoute;