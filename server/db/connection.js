import { MongoClient } from "mongodb";
import dotenv from "dotenv";
dotenv.config();
const client = new MongoClient(process.env.MONGO_URI);
export const db = client.db("ai_mern");
export const connectDB = async () => {
  try {
    await client.connect();
    const uri = new URL(process.env.MONGO_URI);
    const host = uri.hostname;
    console.log(`Database connected on: ${host}`);
  } catch (err) {
    console.error("MongoDB Error:", err);
    process.exit(1);
  }
};