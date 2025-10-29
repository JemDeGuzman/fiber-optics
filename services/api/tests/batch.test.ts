import request from 'supertest';
import app from '../src/app';

describe('Display Existing Batches', () => {
  it('returns { batches: [...] } with expected {id: int,  name str, createdAt: date}', async () => {
    const res = await request(app).get('/api/batches').set('Accept', 'application/json');

    expect(res.status).toBe(200);

    const payload = res.body?.batches ?? res.body;
    expect(Array.isArray(payload)).toBe(true);

    expect(payload).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: 15,
          name: 'Test Batch 2',
          createdAt: '2025-10-26T09:00:59.116Z',
        }),
        expect.objectContaining({
          id: 11,
          name: 'Test Batch 1',
          createdAt: '2025-10-12T02:20:50.456Z',
        }),
      ])
    );
  });
});