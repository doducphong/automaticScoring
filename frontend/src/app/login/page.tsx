'use client';

import React from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  Heading,
  Text,
  Link,
  useToast,
  Card,
  CardBody,
  Flex
} from '@chakra-ui/react';
import { useForm } from 'react-hook-form';
import { useAuth } from '@/contexts/AuthContext';
import NextLink from 'next/link';
import { useRouter } from 'next/navigation';

interface LoginFormData {
  email: string;
  password: string;
}

export default function LoginPage() {
  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormData>();
  const { login, loading } = useAuth();
  const toast = useToast();
  const router = useRouter();

  const onSubmit = async (data: LoginFormData) => {
    try {
      await login(data.email, data.password);
      toast({
        title: 'Login successful',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      router.push('/dashboard');
    } catch (error) {
      toast({
        title: 'Login failed',
        description: error instanceof Error ? error.message : 'Please check your credentials',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <Flex minH="100vh" align="center" justify="center" bg="gray.50">
      <Card maxW="md" w="100%" boxShadow="lg" borderRadius="lg">
        <CardBody p={8}>
          <VStack spacing={6} align="stretch">
            <Box textAlign="center">
              <Heading size="xl" mb={2}>Teacher Scoring System</Heading>
              <Text color="gray.600">Sign in to your account</Text>
            </Box>
            
            <form onSubmit={handleSubmit(onSubmit)}>
              <VStack spacing={4}>
                <FormControl isInvalid={!!errors.email}>
                  <FormLabel htmlFor="email">Email</FormLabel>
                  <Input
                    id="email"
                    type="email"
                    placeholder="Your email"
                    {...register('email', { 
                      required: 'Email is required',
                      pattern: {
                        value: /^\S+@\S+$/i,
                        message: 'Invalid email address',
                      }
                    })}
                  />
                  {errors.email && <Text color="red.500" fontSize="sm">{errors.email.message}</Text>}
                </FormControl>

                <FormControl isInvalid={!!errors.password}>
                  <FormLabel htmlFor="password">Password</FormLabel>
                  <Input
                    id="password"
                    type="password"
                    placeholder="Your password"
                    {...register('password', { 
                      required: 'Password is required',
                      minLength: {
                        value: 6,
                        message: 'Password must be at least 6 characters',
                      }
                    })}
                  />
                  {errors.password && <Text color="red.500" fontSize="sm">{errors.password.message}</Text>}
                </FormControl>

                <Button
                  type="submit"
                  colorScheme="blue"
                  size="lg"
                  width="full"
                  mt={4}
                  isLoading={loading}
                >
                  Sign In
                </Button>
              </VStack>
            </form>

            <Text textAlign="center" mt={4}>
              Don't have an account?{' '}
              <Link as={NextLink} href="/register" color="blue.500">
                Register here
              </Link>
            </Text>
          </VStack>
        </CardBody>
      </Card>
    </Flex>
  );
}
