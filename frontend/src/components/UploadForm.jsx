import React, { useState } from 'react';
import { analyzeVegetation, analyzeSoil } from '../api';
import { motion } from 'framer-motion';
import {
  Upload,
  Image as ImageIcon,
  Leaf,
  Sprout,
  Loader2,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';
import { resizeImage } from '../utils/imageUtils';

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analysis, setAnalysis] = useState('vegetation');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function processImage(file) {
    try {
      const resizedFile = await resizeImage(file);
      setFile(resizedFile);

      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result);
      reader.readAsDataURL(resizedFile);
    } catch (err) {
      console.error('Image processing failed:', err);
      setError('Failed to process image. Please try another file.');
      setFile(null);
      setPreview(null);
    }
  }

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) {
      setFile(null);
      setPreview(null);
      setResult(null);
      setError(null);
      return;
    }

    await processImage(f);
    setResult(null);
    setError(null);
  }

  async function onSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!file) {
      setError('Please select an image.');
      return;
    }

    setLoading(true);
    try {
      if (analysis === 'vegetation') {
        console.log('[Frontend] Submitting vegetation analysis for:', file.name);
        const data = await analyzeVegetation(file);
        console.log('[Frontend] Vegetation result (raw):', data);

        // unwrap if backend wraps response as { status, result }
        const vegetationData = data?.result ?? data;

        setResult({
          type: 'vegetation',
          vegetation: vegetationData?.vegetation ?? false,
          mask_image_base64: vegetationData?.mask_image_base64 ?? null,
          debugInfo: vegetationData?.debug_info ?? null,
        });
      } else {
        console.log('[Frontend] Submitting soil analysis for:', file.name);
        const data = await analyzeSoil(file);
        console.log('[Frontend] Soil result (raw):', data);

        // unwrap if backend wraps response as { status, result }
        const soilData = data?.result ?? data;

        setResult({
          type: 'soil',
          soil_type: soilData?.soil_type ?? 'N/A',
          confidence: Number(soilData?.confidence ?? 0),
        });
      }
    } catch (err) {
      console.error('[Frontend] Analysis failed:', err.response?.data || err);
      setError(err?.response?.data?.error || err.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  async function onDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) {
      await processImage(f);
      setResult(null);
      setError(null);
    }
  }

  function onDragOver(e) {
    e.preventDefault();
  }

  const confidenceBadge = (value) => {
    if (value == null) return null;
    const pct = value * 100;
    let color = 'bg-red-100 text-red-700 border-red-200';
    if (pct >= 80) color = 'bg-green-100 text-green-700 border-green-200';
    else if (pct >= 50) color = 'bg-yellow-100 text-yellow-700 border-yellow-200';
    return (
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${color}`}
      >
        {pct.toFixed(1)}%
      </span>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="bg-white dark:bg-slate-900/60 shadow-soft rounded-2xl p-6 sm:p-8 border border-slate-200 dark:border-slate-700"
    >
      <form onSubmit={onSubmit} className="space-y-5">
        {/* Upload Box */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Upload Image
          </label>
          <div
            onDrop={onDrop}
            onDragOver={onDragOver}
            className="relative flex flex-col items-center justify-center w-full rounded-2xl border-2 border-dashed border-nature-green/40 dark:border-nature-green/30 bg-white dark:bg-slate-900/40 hover:border-nature-green/70 dark:hover:border-nature-green/60 focus-within:border-nature-green/70 transition-shadow shadow-none hover:shadow-soft p-6 cursor-pointer"
          >
            <input
              type="file"
              accept="image/*"
              onChange={onFileChange}
              className="absolute inset-0 opacity-0 cursor-pointer"
            />
            <Upload className="h-6 w-6 text-nature-green mb-2" />
            <p className="text-sm text-slate-600 dark:text-slate-300">
              Drag & drop an image here, or{' '}
              <span className="text-nature-green font-medium">browse</span>
            </p>
          </div>
        </div>

        {/* Analysis Type */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Choose Analysis Type
          </label>
          <select
            value={analysis}
            onChange={(e) => setAnalysis(e.target.value)}
            className="mt-1 block w-full rounded-xl border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900/60 shadow-sm focus:border-nature-green focus:ring-nature-green text-slate-800 dark:text-slate-100"
          >
            <option value="vegetation">Vegetation</option>
            <option value="soil">Soil</option>
          </select>
        </div>

        {/* Submit Button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          type="submit"
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-nature-green px-5 py-2.5 text-white shadow hover:brightness-95 disabled:opacity-60"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Analyzing…
            </>
          ) : (
            <>
              {analysis === 'vegetation' ? (
                <Leaf className="h-4 w-4" />
              ) : (
                <Sprout className="h-4 w-4" />
              )}
              Detect
            </>
          )}
        </motion.button>
      </form>

      {/* Preview */}
      {preview && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6"
        >
          <p className="text-sm text-slate-600 dark:text-slate-300 mb-2 inline-flex items-center gap-1">
            <ImageIcon className="h-4 w-4" /> Preview
          </p>
          <div className="rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-sm bg-white dark:bg-slate-900/40">
            <img
              src={preview}
              alt="preview"
              className="max-h-80 w-full object-contain bg-slate-50 dark:bg-slate-800/40"
            />
          </div>
        </motion.div>
      )}

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 text-sm text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-xl px-3 py-2 inline-flex items-center gap-2"
        >
          <AlertTriangle className="h-4 w-4" /> {error}
        </motion.div>
      )}

      {/* Vegetation Result */}
      {result && result.type === 'vegetation' && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 space-y-3"
        >
          <div className="rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 shadow-sm p-4">
            <div className="flex items-center gap-2 text-slate-800 dark:text-slate-100">
              {result.vegetation ? (
                <CheckCircle2 className="h-5 w-5 text-green-600" />
              ) : (
                <AlertTriangle className="h-5 w-5 text-yellow-600" />
              )}
              <p className="text-sm font-medium">
                {result.vegetation
                  ? 'Vegetation detected'
                  : 'No vegetation detected'}
              </p>
            </div>
            {result.debugInfo && (
              <div className="mt-2 text-xs text-slate-500 dark:text-slate-400 border-t border-slate-200 dark:border-slate-700 pt-2">
                <p>Debug Info:</p>
                <ul className="list-disc list-inside space-y-1">
                  <li>Masks found: {result.debugInfo.has_masks ? 'Yes' : 'No'}</li>
                  <li>
                    Confident boxes:{' '}
                    {result.debugInfo.has_confident_boxes ? 'Yes' : 'No'}
                  </li>
                  <li>Total detections: {result.debugInfo.num_detections}</li>
                  <li>Total masks: {result.debugInfo.num_masks}</li>
                </ul>
              </div>
            )}
          </div>

          {result.mask_image_base64 && (
            <div className="rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 shadow-sm p-4">
              <p className="text-sm text-slate-600 dark:text-slate-300 mb-2">
                Mask overlay
              </p>
              <img
                className="rounded-xl border border-slate-200 dark:border-slate-700"
                src={`data:image/png;base64,${result.mask_image_base64}`}
                alt="mask overlay"
              />
            </div>
          )}
        </motion.div>
      )}

      {/* Soil Result */}
      {result && result.type === 'soil' && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 grid gap-3"
        >
          <div className="rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 shadow-sm p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sprout className="h-5 w-5 text-nature-brown" />
                <p className="text-sm font-medium text-slate-800 dark:text-slate-100">
                  Predicted soil type
                </p>
              </div>
              <p className="text-sm text-slate-700 dark:text-slate-200 font-semibold">
                {result.soil_type || 'N/A'}
              </p>
            </div>
          </div>
          <div className="rounded-2xl bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 shadow-sm p-4">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-slate-800 dark:text-slate-100">
                Confidence
              </p>
              {confidenceBadge(result.confidence || 0)}
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
