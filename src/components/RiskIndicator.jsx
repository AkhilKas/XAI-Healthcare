import React from 'react';

export const RiskIndicator = ({ label, value, color, style }) => (
  <div className="flex flex-col items-center p-4">
    <div className="relative w-20 h-20" style={style}>
      <svg className="transform -rotate-90 w-20 h-20">
        <circle cx="40" cy="40" r="32" stroke="#e5e7eb" strokeWidth="6" fill="none" />
        <circle
          cx="40" cy="40" r="32"
          stroke={color}
          strokeWidth="6"
          fill="none"
          strokeDasharray={`${value * 2.01} 201`}
          className="transition-all duration-1000"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center text-sm font-semibold">
        {value}%
      </div>
    </div>
    <span className="text-xs mt-2 text-gray-600">{label}</span>
  </div>
);