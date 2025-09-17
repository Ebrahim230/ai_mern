import React, { useEffect, useReducer, useRef } from "react";
import { Reload, Rocket, Stop } from "../assets";
import { Chat, New } from "../components";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { setLoading } from "../redux/loading";
import { useDispatch, useSelector } from "react-redux";
import { addList, emptyAllRes, insertNew, livePrompt } from "../redux/messages";
import { emptyUser } from "../redux/user";
import instance from "../config/instance";
import "./style.scss";

const reducer = (state, { type, status }) => {
  switch (type) {
    case "chat":
      return { chat: status, loading: status, resume: status, actionBtns: false };
    case "error":
      return { ...state, error: status };
    case "resume":
      return { chat: true, resume: status, loading: status, actionBtns: true };
    default:
      return state;
  }
};

const Main = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const chatRef = useRef();
  const { user } = useSelector((state) => state);
  const { id = null } = useParams();
  const [status, stateAction] = useReducer(reducer, {
    chat: false,
    error: false,
    actionBtns: false,
  });

  useEffect(() => {
    if (user) {
      dispatch(emptyAllRes());
      setTimeout(() => {
        if (id) {
          const getSaved = async () => {
            try {
              const res = await instance.get("/api/chat/saved", { params: { chatId: id } });
              if (res?.data) {
                dispatch(addList({ _id: id, items: res.data.data }));
                stateAction({ type: "resume", status: false });
              }
            } catch (err) {
              if (err?.response?.data?.status === 404) navigate("/404");
              else alert(err);
            } finally {
              dispatch(setLoading(false));
            }
          };
          getSaved();
        } else {
          stateAction({ type: "chat", status: false });
          dispatch(setLoading(false));
        }
      }, 1000);
    } 
  }, [location]);

  return (
    <div className="main">
      <div className="contentArea">
        {status.chat ? <Chat ref={chatRef} error={status.error} /> : <New />}
      </div>
      <InputArea status={status} chatRef={chatRef} stateAction={stateAction} />
    </div>
  );
};

export default Main;

const InputArea = ({ status, chatRef, stateAction }) => {
  const textAreaRef = useRef();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { prompt, _id } = useSelector((state) => state.messages);

  useEffect(() => {
    textAreaRef.current?.addEventListener("input", () => {
      textAreaRef.current.style.height = "auto";
      textAreaRef.current.style.height = textAreaRef.current.scrollHeight + "px";
    });
  }, []);

  const FormHandle = async () => {
    if (prompt.trim().length === 0) return;
    
    stateAction({ type: "chat", status: true });
    let chatsId = Date.now();
    dispatch(insertNew({ id: chatsId, content: "", prompt }));
    dispatch(livePrompt(""));
    chatRef?.current?.clearResponse();

    try {
      const res = _id 
        ? await instance.put("/api/chat", { chatId: _id, prompt })
        : await instance.post("/api/chat", { prompt });
      
      if (res?.data) {
        const { _id, content } = res.data.data;
        dispatch(insertNew({ _id, fullContent: content, chatsId }));
        chatRef?.current?.loadResponse(stateAction, content, chatsId);
      }
    } catch (err) {
      if (err?.response?.data?.status === 401) {
        dispatch(emptyUser());
        dispatch(emptyAllRes());
        navigate("/login");
      } else {
        stateAction({ type: "error", status: true });
      }
    }
  };

  return (
    <div className="inputArea">
      <div className="flexBody">
        <div className="box">
          <textarea
            placeholder="Ask any question..."
            ref={textAreaRef}
            value={prompt}
            onChange={(e) => dispatch(livePrompt(e.target.value))}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                FormHandle();
              }
            }}
          />
          <button onClick={FormHandle} disabled={status.loading}>{status.loading ? "..." : <Rocket />}</button>
        </div>
      </div>
    </div>
  );
};