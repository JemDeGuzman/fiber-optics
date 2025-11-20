import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import Dashboard from '../src/app/dashboard/page';

jest.mock('@/components/SampleTable', () => ({
  __esModule: true,
  default: ({ samples }: { samples: Array<any> }) => (
    <div data-testid="sample-table">rows:{samples?.length ?? 0}</div>
  ),
}));

jest.mock('@/components/ClassificationPie', () => ({
  __esModule: true,
  default: ({ ratio }: { ratio: Record<string, number> }) => (
    <div data-testid="classification-pie">{JSON.stringify(ratio || {})}</div>
  ),
}));

type FetchResp = { ok: boolean; status: number; json: () => Promise<any>; text?: () => Promise<string> };
const makeResp = (body: any, ok = true, status = 200): FetchResp => ({
  ok,
  status,
  json: async () => body,
  text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
});

function expectFetchUrlCalled(re: RegExp) {
  const calls = (global.fetch as jest.Mock).mock.calls;
  const hit = calls.some((args) => typeof args[0] === 'string' && re.test(String(args[0])));
  expect(hit).toBe(true);
}

beforeEach(() => {
  jest.resetAllMocks();

  (global as any).fetch = jest.fn(async (input: RequestInfo | URL) => {
    const url = String(input);

    if (url.endsWith('/api/auth/me')) {
      return makeResp({}, false, 401);
    }

    if (url.endsWith('/api/batches')) {
      return makeResp({
        batches: [
          { id: 1, name: 'Seed-Generated Batch 1', createdAt: '2025-10-11T14:21:18.567Z' },
          { id: 6, name: 'Seed-Generated Batch 2', createdAt: '2025-10-11T14:21:23.438Z' },
        ],
      });
    }

    if (url.includes('/api/batches/1/samples')) {
      return makeResp({
        samples: [
          { id: 101, label: 'A1' },
          { id: 102, label: 'A2' },
          { id: 103, label: 'A3' },
        ],
        total: 3,
        page: 1,
        limit: 25,
      });
    }

    if (url.includes('/api/batches/1/stats')) {
      return makeResp({ total: 3, ratio: { good: 0.67, bad: 0.33 } });
    }

    return makeResp({}, true, 200);
  }) as jest.Mock;
});

afterEach(() => {
  jest.clearAllMocks();
});

describe('Dashboard data flows', () => {
  it('GET /api/batches successful -> Select Batch Map populated', async () => {
    render(<Dashboard />);

    const opt1 = await screen.findByRole('option', { name: 'Seed-Generated Batch 1' });
    const opt2 = await screen.findByRole('option', { name: 'Seed-Generated Batch 2' });

    const batchPrompt = screen.getByRole('option', { name: /select batch/i });
    const batchSelect = batchPrompt.closest('select') as HTMLSelectElement;
    expect(batchSelect).toBeTruthy();

    expect(screen.getByRole('heading', { name: /welcome to fiber optics!/i })).toBeInTheDocument();

    expectFetchUrlCalled(/\/api\/batches$/);

    expect(batchSelect).toContainElement(opt1);
    expect(batchSelect).toContainElement(opt2);
  });

  it('GET /api/batches/<batchId>/samples and /api/batches/<batchId>/stats successful -> SampleTable and ClassificationPie rendered', async () => {
    const user = userEvent.setup();
    render(<Dashboard />);

    const batchPrompt = await screen.findByRole('option', { name: /select batch/i });
    const batchSelect = batchPrompt.closest('select') as HTMLSelectElement;

    await user.selectOptions(batchSelect, '1');

    await screen.findByTestId('sample-table');
    expect(screen.getByTestId('sample-table')).toHaveTextContent('rows:3');

    await waitFor(() => {
      expect(screen.getByText(/total:\s*3/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Page\s+1\s+of/i)).toBeInTheDocument();

    expectFetchUrlCalled(/\/api\/batches\/1\/samples\?page=1&limit=25/);
    expectFetchUrlCalled(/\/api\/batches\/1\/stats$/);
  });
});