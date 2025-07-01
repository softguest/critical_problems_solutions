import { auth, signOut } from '@/auth';
import React from 'react'
import { Button } from '../ui/button';
import { FaSignOutAlt } from 'react-icons/fa';

const SignOut = async () => {
    const session = await auth();
  return (
    <form action={async() => {
        "use server";
        await signOut();
    }}>
    <Button type="submit" className='rounded-full'>
        <FaSignOutAlt className=''/> <span className='ml-1 hidden md:block'>SignOut</span>
    </Button>
    </form>
  )
}

export default SignOut
