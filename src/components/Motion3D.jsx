import React from 'react';

export const Motion3D = () => (
  <div className="relative w-full h-96 bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg flex items-center justify-center">
    <div className="relative w-64 h-64">
      <svg viewBox="-10 -10 20 20" className="w-full h-full">
        <defs>
          <linearGradient id="armGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3b82f6" />
            <stop offset="100%" stopColor="#1d4ed8" />
          </linearGradient>
          <linearGradient id="torsoGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#22c55e" />
            <stop offset="100%" stopColor="#16a34a" />
          </linearGradient>
        </defs>
        
        <rect x="-2" y="-4" width="4" height="8" fill="url(#torsoGrad)" rx="1" transform="rotate(-15)" />
        <line x1="-2" y1="-2" x2="-6" y2="2" stroke="url(#armGrad)" strokeWidth="1" strokeLinecap="round" />
        <line x1="2" y1="-2" x2="8" y2="-6" stroke="url(#armGrad)" strokeWidth="1.2" strokeLinecap="round" />
        <circle cx="8" cy="-6" r="0.8" fill="#3b82f6" />
        <line x1="-1" y1="4" x2="-2" y2="10" stroke="#6b7280" strokeWidth="1" strokeLinecap="round" />
        <line x1="1" y1="4" x2="2" y2="10" stroke="#6b7280" strokeWidth="1" strokeLinecap="round" />
        <circle cx="0" cy="-5.5" r="1.5" fill="#f59e0b" />
      </svg>
    </div>
    <div className="absolute bottom-4 right-4 bg-white px-3 py-1 rounded text-xs shadow">
      3D Movement Reconstruction
    </div>
  </div>
);