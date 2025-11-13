/**
 * Resizes an image file to 640x640 pixels while preserving aspect ratio through center cropping.
 * Uses canvas for client-side resizing before upload.
 * 
 * @param {File} file - The original image file from input[type="file"]
 * @param {Object} options - Configuration options
 * @param {number} options.targetSize - Target size in pixels (both width and height)
 * @param {string} options.format - Output format ('image/jpeg' or 'image/png')
 * @param {number} options.quality - Image quality (0 to 1), only used for JPEG
 * @returns {Promise<Blob>} A promise that resolves with the resized image as a Blob
 */
export async function resizeImage(file, {
  targetSize = 640,
  format = file.type || 'image/jpeg',
  quality = 0.9
} = {}) {
  return new Promise((resolve, reject) => {
    // Create image to calculate dimensions
    const img = new Image();
    const reader = new FileReader();

    reader.onload = () => {
      img.src = reader.result;
    };

    img.onload = () => {
      // Create canvas for resizing
      const canvas = document.createElement('canvas');
      canvas.width = targetSize;
      canvas.height = targetSize;
      const ctx = canvas.getContext('2d');

      // Determine dimensions to maintain aspect ratio via center crop
      let sourceX = 0;
      let sourceY = 0;
      let sourceWidth = img.width;
      let sourceHeight = img.height;

      if (img.width > img.height) {
        // Landscape: crop width
        sourceWidth = img.height;
        sourceX = (img.width - sourceWidth) / 2;
      } else if (img.height > img.width) {
        // Portrait: crop height
        sourceHeight = img.width;
        sourceY = (img.height - sourceHeight) / 2;
      }

      // Draw image centered and cropped
      ctx.fillStyle = '#FFFFFF'; // White background
      ctx.fillRect(0, 0, targetSize, targetSize);
      ctx.drawImage(
        img,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        0,
        0,
        targetSize,
        targetSize
      );

      // Convert to blob
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error('Canvas to Blob conversion failed'));
            return;
          }
          // Create a new File with the original name but resized content
          const resizedFile = new File([blob], file.name, {
            type: format,
            lastModified: Date.now(),
          });
          resolve(resizedFile);
        },
        format,
        format === 'image/jpeg' ? quality : undefined
      );
    };

    img.onerror = () => reject(new Error('Failed to load image'));
    reader.onerror = () => reject(new Error('Failed to read file'));

    reader.readAsDataURL(file);
  });
}