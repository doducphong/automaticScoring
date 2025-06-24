'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Heading,
  Input,
  Stack,
  Text,
  useToast,
  Avatar,
  VStack,
  InputGroup,
  InputRightElement,
  IconButton
} from '@chakra-ui/react';
import { FiEye, FiEyeOff, FiSave } from 'react-icons/fi';
import api from '@/lib/api';

export default function ProfilePage() {
  const [user, setUser] = useState<any>(null);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });
  const [isLoading, setIsLoading] = useState(false);
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  
  const toast = useToast();

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const res = await api.get('/teachers/me');
        setUser(res.data);
        setFormData(prev => ({
          ...prev,
          name: res.data.name || '',
          email: res.data.email || '',
        }));
      } catch (err) {
        toast({
          title: 'Failed to load profile',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      }
    };
    fetchUser();
  }, [toast]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleUpdateProfile = async () => {
    setIsLoading(true);
    try {
      await api.put('/teachers/me', {
        name: formData.name,
        email: formData.email,
      });

      setUser((prev: any) => ({
        ...prev,
        name: formData.name,
        email: formData.email,
      }));

      toast({
        title: 'Profile updated',
        description: 'Your profile has been updated.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (err: any) {
      toast({
        title: 'Update failed',
        description: err?.response?.data?.detail || 'Failed to update profile.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdatePassword = async () => {
    if (!formData.currentPassword || !formData.newPassword || !formData.confirmPassword) {
      toast({
        title: 'Missing fields',
        description: 'Please fill in all password fields.',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (formData.newPassword !== formData.confirmPassword) {
      toast({
        title: 'Mismatch',
        description: 'Passwords do not match.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsLoading(true);
    try {
      await api.put('/teachers/me', {
        current_password: formData.currentPassword,
        password: formData.newPassword,
      });

      toast({
        title: 'Password updated',
        description: 'Your password has been changed.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });

      setFormData(prev => ({
        ...prev,
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      }));
    } catch (err: any) {
      toast({
        title: 'Password update failed',
        description: err?.response?.data?.detail || 'Failed to update password.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box>
      <Heading size="lg" mb={4}>Profile Settings</Heading>

      <Flex direction={{ base: 'column', md: 'row' }} gap={6}>
        {/* Profile Info */}
        <Card flex={1}>
          <CardHeader>
            <Heading size="md">Personal Information</Heading>
          </CardHeader>
          <Divider />
          <CardBody>
            <VStack spacing={6} align="center" mb={6}>
              <Avatar 
                size="xl" 
                name={user?.name} 
                bg="blue.500"
              />
              <Text fontSize="lg" fontWeight="medium">
                {user?.name}
              </Text>
              <Text color="gray.500">{user?.email}</Text>
            </VStack>

            <Stack spacing={4}>
              <FormControl>
                <FormLabel>Full Name</FormLabel>
                <Input
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Email</FormLabel>
                <Input
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                />
              </FormControl>

              <Button
                leftIcon={<FiSave />}
                colorScheme="blue"
                onClick={handleUpdateProfile}
                isLoading={isLoading}
              >
                Save Changes
              </Button>
            </Stack>
          </CardBody>
        </Card>

        {/* Password Change */}
        <Card flex={1}>
          <CardHeader>
            <Heading size="md">Change Password</Heading>
          </CardHeader>
          <Divider />
          <CardBody>
            <Stack spacing={4}>
              <FormControl>
                <FormLabel>Current Password</FormLabel>
                <InputGroup>
                  <Input
                    name="currentPassword"
                    type={showCurrentPassword ? "text" : "password"}
                    value={formData.currentPassword}
                    onChange={handleInputChange}
                  />
                  <InputRightElement>
                    <IconButton
                      aria-label="Toggle"
                      icon={showCurrentPassword ? <FiEyeOff /> : <FiEye />}
                      size="sm"
                      onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    />
                  </InputRightElement>
                </InputGroup>
              </FormControl>

              <FormControl>
                <FormLabel>New Password</FormLabel>
                <InputGroup>
                  <Input
                    name="newPassword"
                    type={showNewPassword ? "text" : "password"}
                    value={formData.newPassword}
                    onChange={handleInputChange}
                  />
                  <InputRightElement>
                    <IconButton
                      aria-label="Toggle"
                      icon={showNewPassword ? <FiEyeOff /> : <FiEye />}
                      size="sm"
                      onClick={() => setShowNewPassword(!showNewPassword)}
                    />
                  </InputRightElement>
                </InputGroup>
              </FormControl>

              <FormControl>
                <FormLabel>Confirm Password</FormLabel>
                <InputGroup>
                  <Input
                    name="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                  />
                  <InputRightElement>
                    <IconButton
                      aria-label="Toggle"
                      icon={showConfirmPassword ? <FiEyeOff /> : <FiEye />}
                      size="sm"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    />
                  </InputRightElement>
                </InputGroup>
              </FormControl>

              <Button
                colorScheme="blue"
                onClick={handleUpdatePassword}
                isLoading={isLoading}
              >
                Update Password
              </Button>
            </Stack>
          </CardBody>
        </Card>
      </Flex>
    </Box>
  );
}
