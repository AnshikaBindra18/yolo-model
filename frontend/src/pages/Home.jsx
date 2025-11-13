import React from 'react';
import UploadForm from '../components/UploadForm';

export default function Home() {
  return (
    <div className="min-h-[70vh] flex items-start sm:items-center justify-center">
      <div className="w-full max-w-2xl">
        <UploadForm />
      </div>
    </div>
  );
}
