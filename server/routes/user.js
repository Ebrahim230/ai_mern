import { Router } from "express";
import sendMail from "../mail/send.js";
import user from "../helpers/user.js";
import jwt from "jsonwebtoken";
import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config(); 

const router = Router();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CheckLogged = async (req, res) => {
    const token = req.cookies?.userToken;

    if (!token) {
        return res.status(200).json({ status: 200, message: "Not Logged" });
    }

    try {
        const decoded = jwt.verify(token, process.env.JWT_PRIVATE_KEY);
        let userData = await user.checkUserFound(decoded);
        if (!userData) {
            res.clearCookie("userToken");
            return res.status(200).json({ status: 200, message: "Not Logged" });
        }

        delete userData.pass;
        return res.status(200).json({
            status: 200,
            message: "Already Logged",
            data: userData,
        });
    } catch (err) {
        return res.status(200).json({ status: 200, message: "Not Logged" });
    }
};

router.get("/checkLogged", CheckLogged);

// ✅ Signup Route
router.post("/signup", async (req, res) => {
    console.log("🔹 Signup Request Received:", req.body);
    let email = req.body?.email?.toLowerCase();
    if (!email || req.body?.pass?.length < 8) {
        return res.status(422).json({ status: 422, message: "Invalid email or password" });
    }

    try {
        const response = await user.signup(req.body);
        console.log("✅ Signup Response:", response);

        // Read Email Template
        fs.readFile(`${__dirname}/../mail/template.html`, "utf8", (err, html) => {
            if (!err) {
                html = html.replace("[URL]", `${process.env.SITE_URL}/signup/pending/${response._id}`)
                           .replace("[TITLE]", "Verify your email address")
                           .replace("[CONTENT]", "Please verify your email to continue.")
                           .replace("[BTN_NAME]", "Verify email address");

                sendMail({ to: req.body.email, subject: "ThinkestAI - Verify your email", html });
            } else {
                console.log("❌ Error reading email template:", err);
            }
        });

        return res.status(201).json({
            status: 201,
            message: "Success",
            data: { _id: response?._id || null, manual: response?.manual || false },
        });
    } catch (err) {
        console.error("❌ Signup Error:", err);
        return res.status(err?.exists ? 400 : 500).json({
            status: err?.exists ? 400 : 500,
            message: err?.message || "Signup failed",
        });
    }
});

// ✅ Check Pending Signup
router.get("/checkPending", async (req, res) => {
    if (!req.query?._id || req.query._id.length !== 24) {
        return res.status(404).json({ status: 404, message: "Not found" });
    }

    try {
        const response = await user.checkPending(req.query._id);
        return res.status(200).json({ status: 200, message: "Success", data: response });
    } catch (err) {
        return res.status(err?.status || 500).json({ status: err?.status || 500, message: err?.text || "Internal Server Error" });
    }
});

// ✅ Finish Signup
router.put("/signup-finish", async (req, res) => {
    const { _id, fName, lName } = req.body;
    try {
        const response = await user.finishSignup({ _id, fName, lName });
        console.log("Signup finish response:", response);

        if (response) {
            return res.status(200).json({ status: 200, message: "Signup successfully completed", data: response });
        } else {
            return res.status(400).json({ status: 400, message: "Error completing signup" });
        }
    } catch (err) {
        console.error("Error during signup finish:", err);
        return res.status(err?.status || 500).json({ status: err?.status || 500, message: err?.message || "Error finishing signup" });
    }
});

// ✅ Login Route
router.post("/login", async (req, res) => {
    try {
        const response = await user.login(req.body);
        const token = jwt.sign({ _id: response._id, email: response.email }, process.env.JWT_PRIVATE_KEY, { expiresIn: "24h" });

        return res.status(200)
            .cookie("userToken", token, { httpOnly: true, expires: new Date(Date.now() + 86400000) })
            .json({ status: 200, message: "Success", data: response });
    } catch (err) {
        return res.status(422).json({ status: 422, message: "Invalid login credentials!" });
    }
});

// ✅ Logout Route
router.get("/logout", (_, res) => {
    res.clearCookie("userToken").status(200).json({ status: 200, message: "Logged out" });
});

// ✅ Delete Account
router.delete("/account", async (req, res) => {
    try {
        const token = req.cookies?.userToken;
        if (!token) {
            return res.status(200).json({ status: 200, message: "Not Logged" });
        }

        const decoded = jwt.verify(token, process.env.JWT_PRIVATE_KEY);
        const userData = await user.checkUserFound(decoded);
        if (!userData) {
            return res.status(404).json({ status: 404, message: "User not found" });
        }

        await user.deleteUser(userData._id);
        return res.clearCookie("userToken").status(200).json({ status: 200, message: "Account deleted" });
    } catch (err) {
        return res.status(500).json({ status: 500, message: "Error deleting account" });
    }
});

router.post("/forgot-request", async (req, res) => {
    try {
      const response = await user.generateResetToken(req.body.email);
      if (response) {
        const resetLink = `${process.env.SITE_URL}/forgot?token=${response.resetToken}`;
        fs.readFile(`${__dirname}/../mail/template.html`, "utf8", (err, html) => {
          if (!err) {
            html = html.replace("[URL]", resetLink)
                       .replace("[TITLE]", "Reset Password")
                       .replace("[CONTENT]", "Click below to reset your password.")
                       .replace("[BTN_NAME]", "Reset Password");
  
            sendMail({ to: req.body.email, subject: "ThinkestAI - Reset Password", html });
          } else {
            console.log("❌ Error reading email template:", err);
          }
        });
        return res.status(200).json({ status: 200, message: "If the email exists, a reset link has been sent." });
      } else {
        return res.status(200).json({ status: 200, message: "If the email exists, a reset link has been sent." });
      }
    } catch (err) {
      console.error("❌ Forgot Request Error:", err);
      return res.status(500).json({ status: 500, message: "Internal Server Error" });
    }
  });
  
  // ✅ Forgot Password - Reset
  router.put("/forgot-finish", async (req, res) => {
    const { token, newPass, reEnter } = req.body;
    if (newPass !== reEnter) {
      return res.status(400).json({ status: 400, message: "Passwords do not match" });
    }
    try {
      const result = await user.resetPasswordWithToken({ token, newPass });
      return res.status(200).json(result);
    } catch (err) {
      return res.status(err.status || 500).json({ status: err.status || 500, message: err.text || "Error resetting password" });
    }
  });

export default router;
