import {db} from '@/lib/db';
import Link from 'next/link';

const ProblemsPage = async () => {
  const problems = await db.problem.findMany({
    orderBy: {
      createdAt: 'desc',
    },
    include: {
      author: true,
    },
  });

  return (
    <div className='max-w-4xl py-8 mx-4'>
      <h1 className='text-3xl font-bold mb-4'>Problems</h1>
      <div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4'>
        {problems.map((problem) => (
          <Link
            key={problem.id}
            href={`/admin/problem/${problem.id}`}
            className='bg-white p-4 rounded-md shadow-md'
          >
            <h2 className='text-xl font-bold'>{problem.title}</h2>
            <p>Published by: {problem.author?.firstName} {problem.author?.lastName}</p>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default ProblemsPage;
