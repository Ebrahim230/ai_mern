import express from 'express'
import cors from 'cors'
import cookieParser from 'cookie-parser'
import dotenv from 'dotenv'
import path from 'path'
import { fileURLToPath } from 'url'
import ChatRoute from './routes/chat.js'
import UserRoute from './routes/user.js'
import {connectDB}  from './db/connection.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

dotenv.config()
connectDB();
const app = express()
const port = process.env.PORT || 3000

app.use(cors({ credentials: true, origin: process.env.SITE_URL }))
app.use(cookieParser())
app.use(express.json({ limit: '50mb' }))

app.use('/api/chat/', ChatRoute)
app.use('/api/user/', UserRoute)

const reactDistPath = path.join(__dirname, '..', 'client', 'dist')
app.use(express.static(reactDistPath))

app.get('*', (req, res) => {
  res.sendFile(path.join(reactDistPath, 'index.html'))
})

app.listen(port, async()=>{
  console.log(`Server running on port: ${port}`);
})