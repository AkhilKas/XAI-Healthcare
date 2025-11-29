import React from 'react';
import { AlertCircle, CheckCircle, AlertTriangle } from 'lucide-react';

const iconMap = {
  error: AlertCircle,
  warning: AlertTriangle,
  success: CheckCircle,
};

const colorMap = {
  error: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', icon: 'text-red-600' },
  warning: { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', icon: 'text-amber-600' },
  success: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', icon: 'text-green-600' },
};

export const FindingsCard = ({ type, title, description }) => {
  const Icon = iconMap[type];
  const colors = colorMap[type];

  return (
    <div className={`flex items-start gap-3 p-3 ${colors.bg} rounded-lg border ${colors.border}`}>
      <Icon className={`${colors.icon} mt-0.5 flex-shrink-0`} size={18} />
      <div className="text-sm">
        <div className={`font-medium ${colors.text.replace('700', '900')}`}>{title}</div>
        <div className={`${colors.text} text-xs mt-1`}>{description}</div>
      </div>
    </div>
  );
};