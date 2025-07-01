"use server";

import * as z from "zod";
import bcrypt from "bcryptjs";

import { update } from "@/auth";
import { db } from "@/lib/db";
import { SettingsSchema } from "@/schemas";
import { getUserByEmail, getUserById } from "@/data/user";
import { currentUser } from "@/lib/auth";
import { generateVerificationToken } from "@/lib/tokens";
import { sendVerificationEmail } from "@/lib/mail";

export const settings = async (
  values: z.infer<typeof SettingsSchema>
) => {
  const user = await currentUser();

  if (!user) {
    return { error: "Unauthorized" }
  }

  const dbUser = await getUserById(user.id);

  if (!dbUser) {
    return { error: "Unauthorized" }
  }

  if (user.isOAuth) {
    values.profileImage = undefined;
    values.firstName = undefined;
    values.middleName = undefined;
    values.lastName = undefined;
    values.dateOfBirth = undefined;
    values.telePhone = undefined;
    values.gender = undefined;

    values.cityName = undefined;
    values.country = undefined;
    values.pinCode = undefined;

    values.email = undefined;
    values.password = undefined;
    values.newPassword = undefined;
    values.isTwoFactorEnabled = undefined;

    values.typeOfIdentificationCard = undefined;
    values.identificationCardNumber = undefined;
    values.identificationFile = undefined;
  }

  if (values.email && values.email !== user.email) {
    const existingUser = await getUserByEmail(values.email);

    if (existingUser && existingUser.id !== user.id) {
      return { error: "Email already in use!" }
    }

    const verificationToken = await generateVerificationToken(
      values.email
    );
    
    await sendVerificationEmail(
      verificationToken.email,
      verificationToken.token,
    );

    return { success: "Verification email sent!" };
  }

  if (values.password && values.newPassword && dbUser.password) {
    const passwordsMatch = await bcrypt.compare(
      values.password,
      dbUser.password,
    );

    if (!passwordsMatch) {
      return { error: "Incorrect password!" };
    }

    const hashedPassword = await bcrypt.hash(
      values.newPassword,
      10,
    );
    values.password = hashedPassword;
    values.newPassword = undefined;
  }

  const updatedUser = await db.user.update({
    where: { id: dbUser.id },
    data: {
      ...values,
    }
  });

  update({
    user: {
      firstName: updatedUser.firstName as string | undefined,
      middleName: updatedUser.middleName as string | undefined,
      lastName: updatedUser.lastName as string | undefined,
      dateOfBirth: updatedUser.dateOfBirth as string | undefined,
      telePhone:   updatedUser.telePhone as string | undefined,
      gender:      updatedUser.gender as string | undefined,

      cityName:       updatedUser.cityName as string | undefined,
      country:        updatedUser.country as string | undefined,
      pinCode:        updatedUser.pinCode as string | undefined,
  
      isTwoFactorEnabled: updatedUser.isTwoFactorEnabled,

      email: updatedUser.email as string | undefined,
      role: updatedUser.role,

      image: updatedUser.profileImage as string | undefined,
    }
  });

  return { success: "Settings Updated!" }
}