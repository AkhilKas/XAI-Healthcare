import { render, screen } from '@testing-library/react';
import { AnalysisPanel } from '../../../src/components/AnalysisPanel';

test('renders title and all items', () => {
  render(
    <AnalysisPanel
      title="Detailed Analysis"
      color="blue"
      items={['item one', 'item two', 'item three']}
    />
  );
  expect(screen.getByText('Detailed Analysis')).toBeInTheDocument();
  expect(screen.getByText('item one')).toBeInTheDocument();
  expect(screen.getByText('item two')).toBeInTheDocument();
  expect(screen.getByText('item three')).toBeInTheDocument();
});

test('applies color classes by color prop', () => {
  const { container } = render(
    <AnalysisPanel title="t" color="green" items={['a']} />
  );
  expect(container.firstChild).toHaveClass('bg-green-50');
});

test('renders empty when items is empty', () => {
  render(<AnalysisPanel title="empty" color="amber" items={[]} />);
  expect(screen.getByText('empty')).toBeInTheDocument();
});
