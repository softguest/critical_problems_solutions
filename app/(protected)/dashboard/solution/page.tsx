import {db} from '@/lib/db';
import Link from 'next/link';

const SolutionsPage = async () => {
  const solutions = await db.solution.findMany({
    orderBy: {
      createdAt: 'desc',
    },
    include: {
      author: true,
    },
  });

  return (
    <div className='max-w-4xl mx-auto py-8 px-4'>
      <h1 className='text-3xl font-bold mb-4'>Solutions</h1>
      <div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4'>
        {solutions.map((solution) => (
          <Link
            key={solution.id}
            href={`/solution/${solution.id}`}
            className='bg-white p-4 rounded-md shadow-md'
          >
            <h2 className='text-xl font-bold'>{solution.title}</h2>
            <p>Published by: {solution.author?.firstName} {solution.author?.lastName}</p>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default SolutionsPage;
