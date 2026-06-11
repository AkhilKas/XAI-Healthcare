import { render, screen } from '@testing-library/react';
import { InjuryPredictionCard } from '../../../src/components/InjuryPredictionCard';

test('renders default label, status and percentage', () => {
  render(<InjuryPredictionCard />);
  expect(screen.getByText('Injury Prediction')).toBeInTheDocument();
  expect(screen.getByText('Injured')).toBeInTheDocument();
  expect(screen.getByText('80%')).toBeInTheDocument();
});

test('renders custom status and percentage', () => {
  render(<InjuryPredictionCard status="Non-Injured" percentage={25} />);
  expect(screen.getByText('Non-Injured')).toBeInTheDocument();
  expect(screen.getByText('25%')).toBeInTheDocument();
});

test('applies red bar when injured', () => {
  const { container } = render(<InjuryPredictionCard status="Injured" percentage={90} />);
  const bar = container.querySelector('div[style*="background-color"]');
  expect(bar).toHaveStyle({ backgroundColor: '#ef4444' });
});

test('applies green bar when non-injured', () => {
  const { container } = render(<InjuryPredictionCard status="Non-Injured" percentage={10} />);
  const bar = container.querySelector('div[style*="background-color"]');
  expect(bar).toHaveStyle({ backgroundColor: '#22c55e' });
});
