import {db} from '@/lib/db';
import { format } from 'date-fns';
import { FC, JSXElementConstructor, Key, PromiseLikeOfReactNode, ReactElement, ReactNode, ReactPortal } from 'react';

interface SolutionsProps {
  problemId: string;
}
const Solutions: FC<SolutionsProps> = async ({ problemId }) => {
  const solutions = await db.solution.findMany({
    where: {
      problemId,
    },
    include: {
      author: true,
    },
  });

  return (
    <div className='mt-8'>
      <h2 className='text-2xl font-bold'>Solutions</h2>
      <div>
        {solutions.map((solution) => (
          <li key={solution.id} className='mb-4 bg-slate-300 p-2'>
            <div className='flex items-center mb-2'>
              <div className='text-blue-500 font-bold mr-2'>
                {solution.author?.firstName}
              </div>
              <div className='text-gray-500'>
                {format(solution.createdAt, 'MMMM d, yyyy')}
              </div>
            </div>
            <h3 className='text-2xl font-bold'>{solution.title}</h3>
            <p>{solution.content}</p>
          </li>
        ))}
      </div>
    </div>
  );
};

export default Solutions;
