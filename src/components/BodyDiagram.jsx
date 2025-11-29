import React from 'react';

export const BodyDiagram = () => (
  <div className="relative w-full h-64 bg-gray-50 rounded-lg flex items-center justify-center">
    <div className="relative">
      <svg width="120" height="200" viewBox="0 0 120 200">
        <ellipse cx="60" cy="20" rx="15" ry="18" fill="#e5e7eb" />
        <rect x="50" y="35" width="20" height="50" rx="8" fill="#e5e7eb" />
        <line x1="50" y1="45" x2="20" y2="75" stroke="#e5e7eb" strokeWidth="8" strokeLinecap="round" />
        <line x1="70" y1="45" x2="100" y2="75" stroke="#ef4444" strokeWidth="8" strokeLinecap="round" />
        <circle cx="100" cy="75" r="12" fill="#ef4444" opacity="0.3" />
        <line x1="55" y1="85" x2="45" y2="140" stroke="#e5e7eb" strokeWidth="8" strokeLinecap="round" />
        <line x1="65" y1="85" x2="75" y2="140" stroke="#e5e7eb" strokeWidth="8" strokeLinecap="round" />
      </svg>
      <div className="absolute top-8 right-4 bg-red-100 text-red-700 px-3 py-2 rounded-lg text-xs font-medium max-w-xs">
        Jerky motion detected during elevation phase
      </div>
    </div>
  </div>
);