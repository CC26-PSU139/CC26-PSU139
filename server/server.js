import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import cvRoutes from "./src/routes/cv.routes.js";

const app = express();

app.use(cors());
app.use(express.json());

app.use("/api", cvRoutes);

app.get("/", (req, res) => {
  res.json({
    message: "CareerPath API Running",
  });
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});