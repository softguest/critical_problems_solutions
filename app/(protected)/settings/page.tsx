"use client";

import * as z from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useTransition, useState } from "react";
import { useSession } from "next-auth/react";

import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SettingsSchema } from "@/schemas";
import {
  Card,
  CardHeader,
  CardContent,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { settings } from "@/actions/settings";
import {
  Form,
  FormField,
  FormControl,
  FormItem,
  FormLabel,
  FormDescription,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { useCurrentUser } from "@/hooks/use-current-user";
import { FormError } from "@/components/form-error";
import { FormSuccess } from "@/components/form-success";
import { Role } from "@prisma/client";

const SettingsPage = () => {
  const user = useCurrentUser();

  const [error, setError] = useState<string | undefined>();
  const [success, setSuccess] = useState<string | undefined>();
  const { update } = useSession();
  const [isPending, startTransition] = useTransition();

  const form = useForm<z.infer<typeof SettingsSchema>>({
    resolver: zodResolver(SettingsSchema),
    defaultValues: {
      profileImage: user?.profileImage || undefined,
      firstName: user?.firstName || undefined,
      middleName: user?.middleName || undefined,
      lastName: user?.lastName || undefined,
      gender: user?.gender || undefined,
      dateOfBirth: user?.dateOfBirth || undefined,
      telePhone: user?.telePhone || undefined,

      // Address 
      cityName : user?.cityName   || undefined,
      country : user?.country  || undefined,
      pinCode : user?.pinCode  || undefined,

      // Account Information
      email: user?.email || undefined,
      password: undefined,
      newPassword: undefined,
      role: user?.role || undefined,
      isTwoFactorEnabled: user?.isTwoFactorEnabled || undefined,
      
      typeOfIdentificationCard: user?.typeOfIdentificationCard || undefined,
      identificationCardNumber: user?.identificationCardNumber || undefined,
      identificationFile: user?.identificationFile || undefined,
    }
  });

  const onSubmit = (values: z.infer<typeof SettingsSchema>) => {
    startTransition(() => {
      settings(values)
        .then((data) => {
          if (data.error) {
            setError(data.error);
          }

          if (data.success) {
            update();
            setSuccess(data.success);
          }
        })
        .catch(() => setError("Something went wrong!"));
    });
  }

  return ( 
    <div className="w-full md:px-32">
      <Card className="w-full">
        <CardHeader>
          <p className="text-2xl font-semibold text-center">
            ⚙️ Settings.
          </p>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form 
              className="space-y-6" 
              onSubmit={form.handleSubmit(onSubmit)}
            >
              <div className="space-y-4">

                <FormField
                  control={form.control}
                  name="profileImage"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Profile Image</FormLabel>
                      <FormControl>
                        <Input
                          type="file"
                          accept="image/*"
                          disabled={isPending}
                          onChange={async (e) => {
                            const file = e.target.files?.[0];
                            if (!file) return;

                            const formData = new FormData();
                            formData.append("file", file);

                            const res = await fetch("/api/upload/user-upload", {
                              method: "POST",
                              body: formData,
                            });

                            const data = await res.json();

                            if (res.ok) {
                              field.onChange(data.filePath); // Update the form state with the path
                            } else {
                              setError(data.error || "Failed to upload image");
                            }
                          }}
                        />
                      </FormControl>
                      {form.watch("profileImage") && (
                        <img
                          src={form.watch("profileImage")}
                          alt="Profile Preview"
                          className="w-24 h-24 rounded-full mt-2"
                        />
                      )}
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="firstName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>First Name:</FormLabel>
                      <FormControl>
                        <Input
                          {...field}
                          placeholder="First Name"
                          disabled={isPending}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="middleName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Middle Name:</FormLabel>
                      <FormControl>
                        <Input
                          {...field}
                          placeholder="Middle Name"
                          disabled={isPending}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                    control={form.control}
                    name="lastName"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Last Name:</FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            placeholder="Last Name"
                            disabled={isPending}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                />
                <FormField
                    control={form.control}
                    name="dateOfBirth"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>dateOfBirth</FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            // placeholder="john.doe@example.com"
                            type="date"
                            disabled={isPending}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                />
                <FormField
                  control={form.control}
                  name="gender"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Gender</FormLabel>
                      <Select
                        disabled={isPending}
                        onValueChange={field.onChange}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a gender" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="MALE">
                            Male
                          </SelectItem>
                          <SelectItem value="FEMALE">
                            Female
                          </SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                    control={form.control}
                    name="telePhone"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>TelePhone</FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            // placeholder="+237"
                            disabled={isPending}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                />  

                {/* Address  */}

                <FormField
                    control={form.control}
                    name="cityName"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>City Name:</FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            placeholder="City Name"
                            disabled={isPending}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                />
                <FormField
                    control={form.control}
                    name="country"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Country:</FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            placeholder="Country"
                            disabled={isPending}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                />
                <FormField
                    control={form.control}
                    name="pinCode"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Pin Code:</FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            placeholder="pinCode"
                            disabled={isPending}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                />     

                {/* Account Information */}
                {user?.isOAuth === false && (
                  <>
                    <FormField
                      control={form.control}
                      name="email"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Email</FormLabel>
                          <FormControl>
                            <Input
                              {...field}
                              placeholder="john.doe@example.com"
                              type="email"
                              disabled={isPending}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="password"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Password</FormLabel>
                          <FormControl>
                            <Input
                              {...field}
                              placeholder="******"
                              type="password"
                              disabled={isPending}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="newPassword"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>New Password</FormLabel>
                          <FormControl>
                            <Input
                              {...field}
                              placeholder="******"
                              type="password"
                              disabled={isPending}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </>
                )}
              {/* If account is a student accoun? then display  parent datails  */}
                <FormField
                  control={form.control}
                  name="typeOfIdentificationCard"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Type Of Identification Card</FormLabel>
                      <Select
                        disabled={isPending}
                        onValueChange={field.onChange}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a Document" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="National Identification Card">
                              National Identification Card
                          </SelectItem>
                          <SelectItem value="Passport">
                              Passport
                          </SelectItem>
                          <SelectItem value="Drivers Licience">
                              Drivers Licience 
                          </SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                    control={form.control}
                    name="identificationCardNumber"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Identification Card Number</FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            placeholder="identification Card Number"
                            disabled={isPending}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                />
                {/* This will be a file upload field  */}
                <FormField
                    control={form.control}
                    name="identificationFile"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Identification File</FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            placeholder="Identification File "
                            disabled={isPending}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                />
                {/* the next section will be a file display section of the first   */}
                <FormField
                  control={form.control}
                  name="role"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Role</FormLabel>
                      <Select
                        disabled={isPending}
                        onValueChange={field.onChange}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a role" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value={Role.ADMIN}>
                            Admin
                          </SelectItem>
                          <SelectItem value={Role.USER}>
                            User
                          </SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                {user?.isOAuth === false && (
                  <FormField
                    control={form.control}
                    name="isTwoFactorEnabled"
                    render={({ field }) => (
                      <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3 shadow-sm">
                        <div className="space-y-0.5">
                          <FormLabel>Two Factor Authentication</FormLabel>
                          <FormDescription>
                            Enable two factor authentication for your account
                          </FormDescription>
                        </div>
                        <FormControl>
                          <Switch
                            disabled={isPending}
                            checked={field.value}
                            onCheckedChange={field.onChange}
                          />
                        </FormControl>
                      </FormItem>
                    )}
                  />
                )}
              </div>
              <FormError message={error} />
              <FormSuccess message={success} />
              <Button
                disabled={isPending}
                type="submit"
              >
                Save
              </Button>
            </form>
          </Form>
        </CardContent>
      </Card>
    </div>
    
   );
}
 
export default SettingsPage;