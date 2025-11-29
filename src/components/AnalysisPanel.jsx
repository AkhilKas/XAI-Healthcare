import React from 'react';

export const AnalysisPanel = ({ title, color, items }) => {
  const colorMap = {
    blue: 'bg-blue-50 border-blue-200 text-blue-800',
    green: 'bg-green-50 border-green-200 text-green-800',
    amber: 'bg-amber-50 border-amber-200 text-amber-800',
  };

  return (
    <div className={`${colorMap[color]} rounded-lg border p-4`}>
      <h3 className={`text-sm font-semibold mb-3 ${color === 'blue' ? 'text-blue-900' : color === 'green' ? 'text-green-900' : 'text-amber-900'}`}>
        {title}
      </h3>
      <div className="space-y-2 text-xs">
        {items.map((item, idx) => (
          <div key={idx} className="flex items-start gap-2">
            <div className={`w-1.5 h-1.5 rounded-full mt-1.5 ${color === 'blue' ? 'bg-blue-600' : color === 'green' ? 'bg-green-600' : 'bg-amber-600'}`}></div>
            <span>{item}</span>
          </div>
        ))}
      </div>
    </div>
  );
};