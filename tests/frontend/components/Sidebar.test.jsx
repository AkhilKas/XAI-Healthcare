import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { Sidebar } from '../../../src/components/Sidebar';

const renderSidebar = (overrides = {}) => {
  const props = {
    selectedPatient: '',
    setSelectedPatient: vi.fn(),
    selectedTask: '',
    setSelectedTask: vi.fn(),
    onAnalyze: vi.fn(),
    loading: false,
    ...overrides,
  };
  render(<Sidebar {...props} />);
  return props;
};

test('renders patient and task selects with all options', () => {
  renderSidebar();
  // "Select Patient" / "Select Task" appear twice (label + disabled placeholder)
  expect(screen.getAllByText('Select Patient')).toHaveLength(2);
  expect(screen.getAllByText('Select Task')).toHaveLength(2);
  expect(screen.getByRole('option', { name: 'patient_1' })).toBeInTheDocument();
  expect(screen.getByRole('option', { name: 'control_6' })).toBeInTheDocument();
  expect(screen.getByRole('option', { name: 'Jar opening' })).toBeInTheDocument();
  expect(screen.getByRole('option', { name: 'Hammering' })).toBeInTheDocument();
});

test('calls setSelectedPatient when patient changes', () => {
  const { setSelectedPatient } = renderSidebar();
  const select = screen.getAllByRole('combobox')[0];
  fireEvent.change(select, { target: { value: 'patient_3' } });
  expect(setSelectedPatient).toHaveBeenCalledWith('patient_3');
});

test('calls setSelectedTask when task changes', () => {
  const { setSelectedTask } = renderSidebar();
  const select = screen.getAllByRole('combobox')[1];
  fireEvent.change(select, { target: { value: 'Wall cleaning' } });
  expect(setSelectedTask).toHaveBeenCalledWith('Wall cleaning');
});

test('Go button invokes onAnalyze', () => {
  const { onAnalyze } = renderSidebar();
  fireEvent.click(screen.getByRole('button', { name: /Go/i }));
  expect(onAnalyze).toHaveBeenCalled();
});

test('button shows loading text and disabled style when loading', () => {
  renderSidebar({ loading: true });
  const button = screen.getByRole('button');
  expect(button).toHaveTextContent(/Analyzing/i);
  expect(button.className).toContain('opacity-50');
});
