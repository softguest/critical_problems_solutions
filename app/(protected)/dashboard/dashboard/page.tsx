import { auth } from '@/auth';
import FormNewProblem from '@/components/form-new-problem'
import React from 'react'

const Dashboard = async () => {
  const data = await auth();

  return (
    <div>
        <FormNewProblem data={data} />
    </div>
  )
}

export default Dashboard
