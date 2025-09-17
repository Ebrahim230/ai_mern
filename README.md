# AI MERN Application

This repository contains a full-stack **AI-powered MERN application** with both frontend (React) and backend (Node.js/Express), including AI test/train scripts.

## Project Structure

```
root
│
├── client/          # React frontend
│   ├── dev-dist/
│   ├── dist/
│   ├── node_modules/
│   ├── public/
│   ├── src/
│   ├── .env
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── server/          # Node.js/Express backend
│   ├── assets/
│   ├── db/
│   ├── helpers/
│   ├── mail/
│   ├── node_modules/
│   ├── routes/
│   ├── .env
│   ├── app.js
│   ├── package.json
│   ├── package-lock.json
│   ├── responses.json
│   └── user_data
│
└── README.md
```

---

## Client (React)

The client provides a modern UI for interacting with the AI system.

### Features

* User authentication (signup/login/logout)
* Interactive AI interface
* Responsive design for desktop and mobile
* Integration with backend APIs
* Input validation and error handling

### Setup

```bash
cd client
npm install
npm start
```

> Runs at [http://localhost:3000](http://localhost:3000) by default.

---

## Server (Node.js/Express)

The server handles API requests, AI processing, and user management.

### Features

* RESTful API endpoints for AI interaction
* User authentication with JWT
* AI model test/train endpoints
* Database integration (MongoDB)
* Error handling and logging
* Secure admin routes

### Setup

```bash
cd server
npm install
```

Create `.env` file:

```env
PORT=5000
MONGO_URI=your_mongo_connection_string
JWT_SECRET=your_jwt_secret
```

Start the server:

```bash
npm run dev
```

> Runs at [http://localhost:5000](http://localhost:5000) by default.

---

## API Endpoints

| Method | Endpoint             | Description                       |
| ------ | -------------------- | --------------------------------- |
| POST   | `/api/user/register` | Register a new user               |
| POST   | `/api/user/login`    | Login user and return JWT         |
| POST   | `/api/ai/predict`    | Send input to AI model            |
| POST   | `/api/ai/train`      | Train AI model with provided data |
| GET    | `/api/ai/status`     | Check AI model status             |

> Test using Postman or Insomnia.

---

## AI Test/Train Scripts (JSON)

The `data/` folder (or `server/responses.json`) contains sample **test and training datasets** in JSON format.

### Example `train.json`:

```json
[
  { "input": "Hello", "output": "Hi there!" },
  { "input": "How are you?", "output": "I'm doing great, thanks!" }
]
```

### Example `test.json`:

```json
[
  { "input": "What's your name?", "expected_output": "I am your AI assistant." }
]
```

---

## Notes

1. Both client and server require **Node.js 18+**.
2. Make sure **MongoDB** is running locally or via cloud (Atlas).
3. Install dependencies separately in each folder.
4. Keep `.env` files secure (not committed to Git).
5. AI JSON scripts can be tracked or ignored based on preference.

---

## License

MIT License – free to use, modify, and distr