import * as z from "zod";
import { Role } from "@prisma/client";

export const SettingsSchema = z.object({
    profileImage: z.optional(z.string()),
    firstName:    z.optional(z.string()),
    middleName:   z.optional(z.string()),
    lastName:     z.optional(z.string()),
    username:     z.optional(z.string()),
    dateOfBirth:  z.optional(z.string()),
    telePhone:    z.optional(z.string()),
    gender:       z.optional(z.string()),

    // Address
    address: z.optional(z.string()),
    cityName:       z.optional(z.string()),
    country:        z.optional(z.string()),
    pinCode:        z.optional(z.string()), 

    isTwoFactorEnabled: z.optional(z.boolean()),
    
    email:        z.optional(z.string().email()),
    password:     z.optional(z.string().min(6)),
    newPassword:  z.optional(z.string().min(6)),
    role:         z.enum([Role.ADMIN, Role.USER]),
    
    typeOfIdentificationCard: z.optional(z.string()),
    identificationCardNumber: z.optional(z.string()),
    identificationFile:       z.optional(z.string()),
    
})
  .refine((data) => {
    if (data.password && !data.newPassword) {
      return false;
    }

    return true;
  }, {
    message: "New password is required!",
    path: ["newPassword"]
  })
  .refine((data) => {
    if (data.newPassword && !data.password) {
      return false;
    }

    return true;
  }, {
    message: "Password is required!",
    path: ["password"]
  })

export const NewPasswordSchema = z.object({
  password: z.string().min(6, {
    message: "Minimum of 6 characters required",
  }),
});

export const ResetSchema = z.object({
  email: z.string().email({
    message: "Email is required",
  }),
});

export const LoginSchema = z.object({
  email: z.string().email({
    message: "Email is required",
  }),
  password: z.string().min(1, {
    message: "Password is required",
  }),
  code: z.optional(z.string()),
});

export const RegisterSchema = z.object({
  email: z.string().email({
    message: "Email is required",
  }),
  password: z.string().min(6, {
    message: "Minimum 6 characters required",
  }),
  firstName: z.string().min(1, {
    message: "First Name is required",
  }),
  lastName: z.string().min(1, {
    message: "Last Name is required",
  }),
  username: z.string().min(1, {
    message: "username is required",
  }),
});
