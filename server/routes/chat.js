import { Router } from "express";
import dotenv from "dotenv";
import jwt from "jsonwebtoken";
import user from "../helpers/user.js";
import chat from "../helpers/chat.js";
import pythonAI from "../helpers/pythonAI.js";

dotenv.config();

const router = Router();

const CheckUser = async (req, res, next) => {
  try {
    const token = req.cookies?.userToken;
    if (!token) return res.status(401).json({ status: 401, message: "Not Logged In" });
    jwt.verify(token, process.env.JWT_PRIVATE_KEY, async (err, decoded) => {
      if (err) return res.status(403).json({ status: 403, message: "Invalid Token" });
      try {
        const userData = await user.checkUserFound(decoded);
        if (!userData) {
          res.clearCookie("userToken").status(404).json({ status: 404, message: "User Not Found" });
        } else {
          req.body.userId = userData._id;
          next();
        }
      } catch (error) {
        res.status(500).json({ status: 500, message: error.message });
      }
    });
  } catch (error) {
    res.status(500).json({ status: 500, message: error.message });
  }
};

await pythonAI.init();

router.get("/", (req, res) => {
  res.send("Welcome to Self-Trained Chat API v1");
});

router.post("/", CheckUser, async (req, res) => {
  const { prompt, userId } = req.body;
  try {
    if (!prompt || prompt.trim().length === 0) {
      return res.status(400).json({ status: 400, message: "Prompt is required" });
    }
    const responseText = await pythonAI.generateResponse(prompt);
    if (!responseText) throw new Error("Failed to generate response from AI.");
    const chatData = await chat.newResponse(prompt, responseText, userId);
    res.status(200).json({
      status: 200,
      message: "Success",
      data: { _id: chatData.chatId, content: responseText }
    });
  } catch (error) {
    res.status(500).json({ status: 500, message: error.message });
  }
});

router.get("/history", CheckUser, async (req, res) => {
  const { userId } = req.body;
  try {
    const history = await chat.getHistory(userId);
    res.status(200).json({ status: 200, message: "Success", data: history });
  } catch (error) {
    res.status(500).json({ status: 500, message: error.message });
  }
});

router.delete("/all", CheckUser, async (req, res) => {
  const { userId } = req.body;
  try {
    await chat.deleteAllChat(userId);
    res.status(200).json({ status: 200, message: "Success" });
  } catch (error) {
    res.status(500).json({ status: 500, message: error.message });
  }
});

export default router;