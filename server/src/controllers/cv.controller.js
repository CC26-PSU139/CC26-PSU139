import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse');
import { analyzeCVService } from "../services/aiService.js";
import { supabase } from "../config/supabase.js";

export const analyzeCV = async (req, res) => {
  try {
    let cvText = "";

    // Jika ada file PDF yang diupload
    if (req.file) {
      const pdfData = await pdfParse(req.file.buffer);
      cvText = pdfData.text;
    }
    // Jika teks CV dikirim langsung (untuk testing)
    else if (req.body.cv_text) {
      cvText = req.body.cv_text;
    }
    else {
      return res.status(400).json({ error: "CV file atau cv_text diperlukan" });
    }

    if (cvText.trim().length < 50) {
      return res.status(400).json({ error: "CV terlalu pendek atau tidak bisa dibaca" });
    }

    // Simpan log ke Supabase (tidak blocking — error di sini tidak menghentikan proses)
    try {
      await supabase.from("cv_analysis_logs").insert([{
        cv_preview: cvText,
        created_at: new Date().toISOString(),
      }]);
    } catch (supabaseErr) {
      console.warn("Supabase log error (non-blocking):", supabaseErr.message);
    }

    // Kirim ke FastAPI
    const result = await analyzeCVService(cvText);
    return res.json(result);

  } catch (error) {
    console.error("analyzeCV error:", error.message);

    if (error.code === "ECONNABORTED") {
      return res.status(504).json({ error: "Model AI sedang sibuk, coba lagi" });
    }

    return res.status(500).json({ error: "Internal Server Error" });
  }
};