import dotenv from "dotenv";
dotenv.config();

import axios from "axios";

const AI_SERVICE_URL = process.env.AI_SERVICE_URL;
console.log("AI_SERVICE_URL:", AI_SERVICE_URL);

export const analyzeCVService = async (cvText) => {
  const response = await axios.post(
    `${AI_SERVICE_URL}/predict`,
    { cv_text: cvText },
    { timeout: 60000 }
  );
  return response.data;
};